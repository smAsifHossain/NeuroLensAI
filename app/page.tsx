"use client";

import { ChangeEvent, DragEvent, useEffect, useMemo, useRef, useState } from "react";

type RunStage = "idle" | "ready" | "connecting" | "uploading" | "queued" | "processing" | "complete" | "error";
type ServiceState = "checking" | "online" | "offline" | "unconfigured";
type RegionMetric = {
  voxel_count: number;
  volume_ml: number;
};

type AnalysisManifest = {
  schema_version: string;
  case_id: string;
  summary: string;
  research_only: boolean;
  validation: {
    shape: number[];
    spacing_mm: number[];
    orientation: string[];
    warnings: string[];
  };
  inference: {
    model_id: string;
    elapsed_seconds: number;
    peak_gpu_memory_mb: number | null;
    model_sha256: string;
    mask_sha256: string;
    hardware: string | null;
  };
  measurements: {
    regions: {
      whole_tumor: RegionMetric;
      tumor_core: RegionMetric;
      enhancing_tumor: RegionMetric;
      flair_abnormality_outside_core: RegionMetric;
    };
    spatial_summary: {
      laterality: string;
    };
  };
  evaluation: {
    dice: Record<string, number>;
  } | null;
};

type AnalysisResult = {
  manifest: AnalysisManifest;
  comparisonUrl: string;
  maskUrl: string;
  reportUrl: string;
};

type RemoteFile = {
  url?: string;
  path?: string;
};

const MAX_CASE_BYTES = 1.5 * 1024 * 1024 * 1024;
const architectureStages = [
  {
    id: "input",
    step: "01",
    label: "Multimodal input",
    title: "Four MRI sequences become four input channels.",
    detail: "T1c, T1, T2, and FLAIR volumes are loaded together so the model can use complementary tissue information from the same scan.",
    specs: ["4 input channels", "3D NIfTI volumes"],
  },
  {
    id: "normalize",
    step: "02",
    label: "Preprocessing",
    title: "Each MRI sequence is normalized separately.",
    detail: "Only non-zero intensity values are normalized, channel by channel. This keeps background voxels out of the intensity calculation.",
    specs: ["Non-zero voxels", "Channel-wise normalization"],
  },
  {
    id: "encoder",
    step: "03",
    label: "Encoder",
    title: "Residual blocks learn features at four scales.",
    detail: "The encoder begins with 16 filters and progressively compresses the 3D scan. Residual connections help preserve useful feature information through deeper blocks.",
    specs: ["16 initial filters", "Down blocks 1 · 2 · 2 · 4"],
  },
  {
    id: "bottleneck",
    step: "04",
    label: "Bottleneck",
    title: "The deepest layer combines the scan context.",
    detail: "This compact representation connects large-scale spatial context with the detailed features carried forward by encoder skip connections.",
    specs: ["3D residual features", "Dropout 0.2"],
  },
  {
    id: "decoder",
    step: "05",
    label: "Decoder",
    title: "The model reconstructs a full-resolution prediction.",
    detail: "Three upsampling stages combine deep features with matching encoder features, restoring spatial detail for the final tumor-region prediction.",
    specs: ["Up blocks 1 · 1 · 1", "Encoder skip connections"],
  },
  {
    id: "mask",
    step: "06",
    label: "Predicted mask",
    title: "Three probability channels become one labeled mask.",
    detail: "Sigmoid outputs for tumor core, whole tumor, and enhancing tumor are thresholded at 0.5 and converted into the final NIfTI segmentation mask.",
    specs: ["3 output channels", "0.5 threshold"],
  },
  {
    id: "review",
    step: "07",
    label: "Review",
    title: "The mask powers every result shown in NeuroLens AI.",
    detail: "Overlays, region volumes, spatial measurements, and the PDF findings are generated directly from the predicted mask. No VLM is used for report writing.",
    specs: ["4 predicted-mask overlays", "Measurements · NIfTI · PDF"],
  },
] as const;

const stageCopy: Record<RunStage, { title: string; detail: string }> = {
  idle: { title: "Choose an MRI scan ZIP", detail: "Include T1c, T1, T2, and FLAIR NIfTI files in one ZIP." },
  ready: { title: "MRI scan ready", detail: "Your files are ready for brain tumor segmentation." },
  connecting: { title: "Connecting to the AI service", detail: "The GPU may need a few seconds to wake up." },
  uploading: { title: "Uploading your MRI scan", detail: "Keep this page open while the ZIP is transferred." },
  queued: { title: "Waiting for the GPU", detail: "Your scan is next in the processing queue." },
  processing: { title: "Running the SegResNet model", detail: "The model is creating tumor masks across all four MRI sequences." },
  complete: { title: "Segmentation complete", detail: "Review the visual comparison, measurements, mask, and PDF summary below." },
  error: { title: "Analysis could not finish", detail: "Read the message below and try again." },
};

function formatBytes(bytes: number) {
  if (bytes < 1024 * 1024) return `${Math.max(1, Math.round(bytes / 1024))} KB`;
  if (bytes < 1024 * 1024 * 1024) return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
  return `${(bytes / (1024 * 1024 * 1024)).toFixed(2)} GB`;
}

function normalizeWorkerUrl(value: string) {
  const trimmed = value.trim().replace(/\/$/, "");
  if (!trimmed) return "";
  const parsed = new URL(trimmed);
  if (!/^https?:$/.test(parsed.protocol)) throw new Error("Use an http or https worker URL.");
  return parsed.toString().replace(/\/$/, "");
}

function remoteUrl(value: unknown, workerUrl: string) {
  const candidate = typeof value === "string" ? value : (value as RemoteFile | null)?.url;
  if (!candidate) return "";
  return new URL(candidate, `${workerUrl}/`).toString();
}

function parseManifest(value: unknown) {
  return (typeof value === "string" ? JSON.parse(value) : value) as AnalysisManifest;
}

export default function Home() {
  const fileInputRef = useRef<HTMLInputElement>(null);
  const uploadRef = useRef<HTMLElement>(null);
  const [selectedFile, setSelectedFile] = useState<File | null>(null);
  const [stage, setStage] = useState<RunStage>("idle");
  const [error, setError] = useState("");
  const [isDragging, setIsDragging] = useState(false);
  const [workerUrl, setWorkerUrl] = useState("");
  const [serviceState, setServiceState] = useState<ServiceState>("checking");
  const [connectionCheck, setConnectionCheck] = useState(0);
  const [result, setResult] = useState<AnalysisResult | null>(null);
  const [activeArchitectureStage, setActiveArchitectureStage] = useState<(typeof architectureStages)[number]["id"]>("input");

  useEffect(() => {
    const hydrationTimer = window.setTimeout(() => {
      void (async () => {
        let managed = "";
        const configUrls = [
          "/api/worker-config",
          new URL("worker-config.json", document.baseURI).toString(),
        ];
        for (const configUrl of configUrls) {
          try {
            const response = await fetch(configUrl, { cache: "no-store" });
            if (response.ok) {
              const config = await response.json() as { workerUrl?: string };
              managed = config.workerUrl || "";
              if (managed) break;
            }
          } catch {
            managed = "";
          }
        }

        if (managed) {
          try {
            const normalized = normalizeWorkerUrl(managed);
            setWorkerUrl(normalized);
          } catch {
            setServiceState("unconfigured");
          }
        } else {
          setServiceState("unconfigured");
        }
      })();
    }, 0);
    return () => window.clearTimeout(hydrationTimer);
  }, []);

  useEffect(() => {
    let cancelled = false;
    if (!workerUrl) return;

    void (async () => {
      setServiceState("checking");
      try {
        const { Client } = await import("@gradio/client");
        await Client.connect(workerUrl);
        if (!cancelled) setServiceState("online");
      } catch {
        if (!cancelled) setServiceState("offline");
      }
    })();

    return () => { cancelled = true; };
  }, [workerUrl, connectionCheck]);

  const status = stageCopy[stage];
  const architectureStage = architectureStages.find((item) => item.id === activeArchitectureStage) ?? architectureStages[0];
  const isRunning = ["connecting", "uploading", "queued", "processing"].includes(stage);
  const serviceLabel: Record<ServiceState, string> = {
    checking: "CHECKING GPU",
    online: "GPU ONLINE",
    offline: "GPU OFFLINE",
    unconfigured: "GPU NOT CONFIGURED",
  };
  const regionRows = useMemo(() => {
    if (!result) return [];
    const regions = result.manifest.measurements.regions;
    return [
      ["Whole tumor", regions.whole_tumor, "Violet"],
      ["Tumor core", regions.tumor_core, "Mint"],
      ["Enhancing tumor", regions.enhancing_tumor, "Coral"],
      ["Peritumoral edema", regions.flair_abnormality_outside_core, "Indigo"],
    ] as const;
  }, [result]);

  const scrollToUpload = () => {
    uploadRef.current?.scrollIntoView({ behavior: "smooth", block: "start" });
  };

  const chooseFile = (file?: File) => {
    setError("");
    setResult(null);
    if (!file) {
      setSelectedFile(null);
      setStage("idle");
      return;
    }
    if (!file.name.toLowerCase().endsWith(".zip")) {
      setSelectedFile(null);
      setStage("error");
      setError("Choose one .zip archive containing T1c, T1, T2, and FLAIR MRI files.");
      return;
    }
    if (file.size > MAX_CASE_BYTES) {
      setSelectedFile(null);
      setStage("error");
      setError("The selected ZIP is larger than the 1.5 GB research limit.");
      return;
    }
    setSelectedFile(file);
    setStage("ready");
  };

  const handleInput = (event: ChangeEvent<HTMLInputElement>) => {
    chooseFile(event.target.files?.[0]);
    event.target.value = "";
  };

  const handleDrop = (event: DragEvent<HTMLDivElement>) => {
    event.preventDefault();
    setIsDragging(false);
    chooseFile(event.dataTransfer.files?.[0]);
  };

  const resetCase = () => {
    setSelectedFile(null);
    setResult(null);
    setError("");
    setStage("idle");
    fileInputRef.current?.click();
  };

  const runAnalysis = async () => {
    if (!selectedFile) return;
    if (!workerUrl) {
      setStage("error");
      setError("The inference service is not configured. A project operator must add a stable worker endpoint.");
      return;
    }

    setError("");
    setResult(null);
    try {
      setStage("connecting");
      setServiceState("checking");
      const { Client, handle_file } = await import("@gradio/client");
      const app = await Client.connect(workerUrl, { events: ["data", "status"] });
      setServiceState("online");
      setStage("uploading");
      const submission = app.submit("/analyze", [handle_file(selectedFile)]);
      let finalData: unknown[] | null = null;

      for await (const message of submission) {
        if (message.type === "status") {
          if (message.stage === "pending") setStage("queued");
          if (message.stage === "generating") setStage("processing");
          if (message.stage === "error") throw new Error(message.message || "The GPU worker reported an error.");
        }
        if (message.type === "data") finalData = message.data as unknown[];
      }

      if (!finalData || finalData.length < 4) throw new Error("The worker returned an incomplete result.");
      const manifest = parseManifest(finalData[0]);
      const nextResult = {
        manifest,
        comparisonUrl: remoteUrl(finalData[1], workerUrl),
        maskUrl: remoteUrl(finalData[2], workerUrl),
        reportUrl: remoteUrl(finalData[3], workerUrl),
      };
      if (!nextResult.comparisonUrl || !nextResult.maskUrl || !nextResult.reportUrl) {
        throw new Error("The worker completed inference but did not return all result files.");
      }
      setResult(nextResult);
      setStage("complete");
      window.setTimeout(() => document.getElementById("analysis-result")?.scrollIntoView({ behavior: "smooth", block: "start" }), 50);
    } catch (analysisError) {
      setServiceState("offline");
      setStage("error");
      setError(analysisError instanceof Error ? analysisError.message : "The case could not be processed.");
    }
  };

  return (
    <main>
      <header className="topbar">
        <a className="brand" href="#top" aria-label="NeuroLens AI home">
          <span className="brandMark" aria-hidden="true"><i /><i /><i /></span>
          <span>NeuroLens <b>AI</b></span>
        </a>
        <nav aria-label="Primary navigation">
          <a href="#architecture">Architecture</a>
          <a href="#performance">Performance</a>
          <a className="developerCredit" href="https://www.linkedin.com/in/smasifhossain" target="_blank" rel="noreferrer" aria-label="Developed by S M Asif Hossain — open LinkedIn profile">
            <span>Developed by S M Asif Hossain</span><i aria-hidden="true">in</i>
          </a>
          <button onClick={scrollToUpload}>Upload MRI scan</button>
        </nav>
      </header>

      <section className="hero" id="top">
        <div className="heroCopy">
          <p className="eyebrow"><span /> AI-POWERED BRAIN MRI SEGMENTATION</p>
          <h1>Explore brain tumor regions.<br /><em>Across every MRI sequence.</em></h1>
          <p className="heroDescription">
            NeuroLens AI uses the SegResNet model to analyze T1c, T1, T2, and FLAIR MRI volumes. It returns color-coded tumor regions, volume measurements, a NIfTI mask, and a downloadable PDF summary.
          </p>
          <div className="heroActions">
            <button className="primaryButton" onClick={() => { scrollToUpload(); window.setTimeout(() => fileInputRef.current?.click(), 450); }}>
              Upload MRI scan ZIP <span aria-hidden="true">&rarr;</span>
            </button>
            <a href="#workflow">Read the protocol</a>
          </div>
          <p className="privacyNote"><span aria-hidden="true">&loz;</span> Use de-identified MRI scans only. This tool does not provide a diagnosis.</p>
        </div>

        <div className="heroProcess" aria-label="Upload-to-result sequence">
          <div className="processHeader"><span>RUN PROTOCOL / 001</span><span className={`servicePill ${serviceState}`}><i /> {serviceLabel[serviceState]}</span></div>
          <div className="processTrack" aria-hidden="true">
            <div><span>01</span><i className="archiveIcon">ZIP</i><strong>Upload</strong></div>
            <b>&rarr;</b>
            <div><span>02</span><i className="modelIcon"><em /><em /><em /></i><strong>Segment</strong></div>
            <b>&rarr;</b>
            <div><span>03</span><i className="resultIcon"><em /><em /><em /></i><strong>Review</strong></div>
          </div>
          <p>One upload produces a five-view comparison, tumor measurements, a NIfTI mask, and a PDF summary.</p>
        </div>
      </section>

      <section className="editionStrip" aria-label="Workflow facts">
        <div><small>INPUT</small><strong>MRI scan ZIP</strong></div>
        <div><small>SEQUENCES</small><strong>T1c · T1 · T2 · FLAIR</strong></div>
        <div><small>MODEL</small><strong>SegResNet</strong></div>
        <div><small>OUTPUT</small><strong>Mask · measurements · PDF</strong></div>
      </section>

      <section className="uploadWorkspace" ref={uploadRef} aria-labelledby="upload-title">
        <div className="sectionHeading">
          <p className="eyebrow">UPLOAD MRI SCAN / 01</p>
          <h2 id="upload-title">Upload your multimodal MRI scan ZIP.</h2>
          <p>Include one T1c, T1, T2, and FLAIR NIfTI file. NeuroLens AI checks the files before starting the segmentation.</p>
        </div>

        <div className="uploadGrid">
          <div
            className={`caseDropzone ${isDragging ? "dragging" : ""} ${selectedFile ? "hasFile" : ""}`}
            onDragEnter={(event) => { event.preventDefault(); setIsDragging(true); }}
            onDragOver={(event) => event.preventDefault()}
            onDragLeave={() => setIsDragging(false)}
            onDrop={handleDrop}
          >
            <input ref={fileInputRef} className="visuallyHidden" type="file" accept=".zip,application/zip" onChange={handleInput} />
            <div className="uploadStatusRow">
              <span className={`statusDot ${stage}`} />
              <div><small>CURRENT STATUS</small><strong>{status.title}</strong></div>
            </div>

            {!selectedFile ? (
              <div className="dropPrompt">
                <div className="uploadGlyph" aria-hidden="true"><span>ZIP</span></div>
                <h3>Drop your MRI scan ZIP here</h3>
                <p>{status.detail}</p>
                <button className="primaryButton" onClick={() => fileInputRef.current?.click()}>Choose MRI scan ZIP</button>
              </div>
            ) : (
              <div className="selectedCase">
                <div className="fileIdentity">
                  <span className="zipBadge">ZIP</span>
                  <div><small>SELECTED MRI SCAN</small><h3>{selectedFile.name.replace(/\.zip$/i, "")}</h3><p>{formatBytes(selectedFile.size)} • one ZIP archive</p></div>
                </div>
                <div className="selectedActions">
                  <button className="primaryButton" onClick={runAnalysis} disabled={isRunning || stage === "complete"}>
                    {isRunning ? "Segmentation in progress…" : stage === "complete" ? "Segmentation complete" : "Run brain tumor segmentation"}
                  </button>
                  <button className="textButton" onClick={resetCase} disabled={isRunning}>Choose another ZIP</button>
                </div>
                {isRunning && <div className="progressBar" aria-label="Analysis in progress"><i /></div>}
                <p className="stageDetail">{status.detail}</p>
              </div>
            )}

            {error && <p className="errorMessage" role="alert">{error}</p>}
          </div>

          <aside className="inputContract" aria-label="Required MRI scan files">
            <p className="eyebrow">WHAT TO INCLUDE</p>
            <h3>One scan from each MRI sequence.</h3>
            <div className="modalityList">
              <div><span>01</span><strong>T1c</strong><small>contrast-enhanced T1</small></div>
              <div><span>02</span><strong>T1</strong><small>native T1</small></div>
              <div><span>03</span><strong>T2</strong><small>T2-weighted</small></div>
              <div><span>04</span><strong>FLAIR</strong><small>fluid-sensitive</small></div>
            </div>
            <ul>
              <li>3D NIfTI files: .nii or .nii.gz</li>
              <li>All four files must cover the same brain scan and have matching dimensions</li>
              <li>Voxel spacing close to 1 × 1 × 1 mm is recommended</li>
            </ul>
          </aside>
        </div>

        <details className="workerSetup" open={serviceState !== "online"}>
          <summary><span>Managed inference service</span><strong>{serviceLabel[serviceState]}</strong></summary>
          <div>
            <p>The app connects to its managed GPU automatically. If Modal has paused an idle container, the first check wakes it while this endpoint remains unchanged.</p>
            <button className="connectionRetry" onClick={() => setConnectionCheck((value) => value + 1)}>Check connection again</button>
          </div>
        </details>
      </section>

      {result && (
        <section className="resultSection" id="analysis-result" aria-labelledby="result-title">
          <div className="resultHeading">
            <div><p className="eyebrow">SEGMENTATION RESULT / 02</p><h2 id="result-title">Results for {result.manifest.case_id}</h2></div>
            <span className="resultVerified"><i /> GENERATED FROM THIS UPLOAD</span>
          </div>

          <div className="resultGrid">
            <section className="viewerCard" aria-labelledby="viewer-title">
              <div className="panelHeader"><div><span>01</span><h3 id="viewer-title">Tumor segmentation across MRI sequences</h3></div><small>ONE ORIGINAL + FOUR AI OVERLAYS</small></div>
              <div className="actualScan">
                {/* The worker returns a temporary runtime URL, so framework image optimization cannot preconfigure its host. */}
                {/* eslint-disable-next-line @next/next/no-img-element */}
                <img src={result.comparisonUrl} alt="Original FLAIR reference and predicted tumor mask overlays on T1c, T1, T2, and FLAIR" />
              </div>
              <div className="legend">
                <span><i className="indigo" /> Peritumoral edema</span>
                <span><i className="mint" /> Enhancing tumor</span>
                <span><i className="coral" /> Necrotic or non-enhancing core</span>
              </div>
            </section>

            <aside className="resultDetails">
              <section className="measurementsCard" aria-labelledby="measurements-title">
                <div className="panelHeader"><div><span>02</span><h3 id="measurements-title">Measured regions</h3></div></div>
                <div className="primaryVolume"><small>WHOLE TUMOR</small><strong>{result.manifest.measurements.regions.whole_tumor.volume_ml.toFixed(3)} <em>mL</em></strong><p>{result.manifest.measurements.spatial_summary.laterality}</p></div>
                <div className="measurementRows">
                  {regionRows.slice(1).map(([label, metric, color]) => <div key={label}><i className={color.toLowerCase()} /><span>{label}</span><strong>{metric.volume_ml.toFixed(3)} mL</strong></div>)}
                </div>
              </section>

              <section className="reportCard" aria-labelledby="report-title">
                <div className="panelHeader"><div><span>03</span><h3 id="report-title">Automated segmentation findings</h3></div><small>MEASUREMENT-BASED · NO VLM</small></div>
                <p>{result.manifest.summary}</p>
                {result.manifest.evaluation?.dice && (
                  <div className="diceLine"><small>SUPPLIED-LABEL EVALUATION</small><span>{Object.entries(result.manifest.evaluation.dice).map(([key, value]) => `${key.replaceAll("_", " ")} ${value.toFixed(5)}`).join(" • ")}</span></div>
                )}
                <div className="downloadRow">
                  <a href={result.reportUrl} download>Download PDF summary</a>
                  <a href={result.maskUrl} download>Download NIfTI mask</a>
                </div>
              </section>
            </aside>
          </div>

          <section className="provenanceCard" aria-labelledby="provenance-title">
            <div><p className="eyebrow">PROCESSING DETAILS / 03</p><h3 id="provenance-title">How this result was produced.</h3></div>
            <dl>
              <div><dt>Model</dt><dd>SegResNet · bundle version 0.5.4</dd></div>
              <div><dt>Runtime</dt><dd>{result.manifest.inference.elapsed_seconds.toFixed(3)} seconds</dd></div>
              <div><dt>Hardware</dt><dd>{result.manifest.inference.hardware || "GPU worker"}</dd></div>
              <div><dt>Peak GPU memory</dt><dd>{result.manifest.inference.peak_gpu_memory_mb ? `${result.manifest.inference.peak_gpu_memory_mb.toLocaleString()} MB` : "Not reported"}</dd></div>
              <div><dt>Input geometry</dt><dd>{result.manifest.validation.shape.join(" × ")} • {result.manifest.validation.spacing_mm.join(" × ")} mm</dd></div>
              <div><dt>Mask SHA-256</dt><dd>{result.manifest.inference.mask_sha256}</dd></div>
            </dl>
          </section>
        </section>
      )}

      <section className="workflow" id="workflow" aria-labelledby="workflow-title">
        <div className="workflowIntro"><p className="eyebrow">HOW IT WORKS / 04</p><h2 id="workflow-title">From multimodal MRI files to brain tumor segmentation.</h2><p>NeuroLens AI checks the scan, normalizes each MRI sequence, runs the SegResNet model on the GPU, and calculates measurements from the predicted mask.</p></div>
        <div className="workflowSteps">
          <article><span>01</span><div><small>CHECK FILES</small><h3>Verify</h3><p>Confirm the ZIP contains matching T1c, T1, T2, and FLAIR NIfTI volumes.</p></div></article>
          <article><span>02</span><div><small>PREPROCESS</small><h3>Normalize</h3><p>Normalize non-zero intensity values separately within each MRI sequence.</p></div></article>
          <article><span>03</span><div><small>AI MODEL</small><h3>Segment</h3><p>The SegResNet model creates tumor-region predictions with GPU sliding-window inference.</p></div></article>
          <article><span>04</span><div><small>RESULTS</small><h3>Review</h3><p>Compare all four overlays, review measurements, and download the mask and PDF summary.</p></div></article>
        </div>
      </section>

      <section className="architecture" id="architecture" aria-labelledby="architecture-title">
        <div className="architectureIntro">
          <p className="eyebrow">INTERACTIVE MODEL VIEW / 05</p>
          <h2 id="architecture-title">Explore the SegResNet architecture and processing pipeline.</h2>
          <p>Select or hover over a stage to see how four MRI sequences move through preprocessing, the 3D encoder-decoder network, mask creation, and the final review.</p>
        </div>

        <div className="architectureShell">
          <div className="architectureRail" role="tablist" aria-label="SegResNet processing stages">
            {architectureStages.map((item, index) => (
              <button
                key={item.id}
                id={`architecture-tab-${item.id}`}
                className={item.id === architectureStage.id ? "active" : ""}
                role="tab"
                aria-selected={item.id === architectureStage.id}
                aria-controls="architecture-panel"
                onClick={() => setActiveArchitectureStage(item.id)}
                onMouseEnter={() => setActiveArchitectureStage(item.id)}
                onFocus={() => setActiveArchitectureStage(item.id)}
              >
                <span>{item.step}</span>
                <strong>{item.label}</strong>
                {index < architectureStages.length - 1 && <i aria-hidden="true" />}
              </button>
            ))}
          </div>

          <div
            className={`architecturePanel stage-${architectureStage.id}`}
            id="architecture-panel"
            role="tabpanel"
            aria-live="polite"
            aria-labelledby={`architecture-tab-${architectureStage.id}`}
          >
            <div className="networkDiagram" aria-hidden="true">
              <div className="networkStack inputStack"><i /><i /><i /><i /></div>
              <span className="networkLink" />
              <div className="networkStack encoderStack"><i /><i /><i /><i /></div>
              <div className="skipConnections"><i /><i /><i /></div>
              <div className="networkCore"><i /><i /><i /></div>
              <div className="networkStack decoderStack"><i /><i /><i /></div>
              <span className="networkLink" />
              <div className="networkMask"><i /><i /><i /></div>
            </div>
            <div className="architectureCopy" key={architectureStage.id}>
              <small>STAGE {architectureStage.step} · {architectureStage.label}</small>
              <h3>{architectureStage.title}</h3>
              <p>{architectureStage.detail}</p>
              <div>{architectureStage.specs.map((spec) => <span key={spec}>{spec}</span>)}</div>
            </div>
          </div>
        </div>
      </section>

      <section className="performance" id="performance" aria-labelledby="performance-title">
        <div className="performanceIntro">
          <div>
            <p className="eyebrow">PUBLISHED MODEL EVALUATION / 06</p>
            <h2 id="performance-title">SegResNet performance, with the evaluation context attached.</h2>
          </div>
          <div className="benchmarkSource">
            <span>Bundle version 0.5.4</span>
            <p>Author-reported validation results for the model weights used by NeuroLens AI. These are benchmark results, not a score for an uploaded scan.</p>
            <a href="https://huggingface.co/MONAI/brats_mri_segmentation/blob/main/docs/README.md#performance" target="_blank" rel="noreferrer">View the published benchmark source <b aria-hidden="true">↗</b></a>
          </div>
        </div>

        <div className="diceHeader">
          <div><small>EVALUATION METRIC</small><strong>Dice score</strong></div>
          <p>Dice measures overlap between the predicted mask and the reference label. It ranges from 0 to 1; higher is better, and 1 means perfect overlap.</p>
        </div>

        <div className="metricGrid" aria-label="Published validation Dice scores">
          {[
            ["Average", "0.8518", "85.18%", "All three target regions"],
            ["Whole tumor", "0.9026", "90.26%", "Complete predicted tumor extent"],
            ["Tumor core", "0.8559", "85.59%", "Core including enhancing region"],
            ["Enhancing tumor", "0.7905", "79.05%", "Contrast-enhancing region"],
          ].map(([label, score, percentage, detail], index) => (
            <article className={index === 0 ? "averageMetric" : ""} key={label}>
              <span>{String(index + 1).padStart(2, "0")}</span>
              <small>{label}</small>
              <strong>{score}</strong>
              <div className="metricTrack" aria-hidden="true"><i style={{ width: percentage }} /></div>
              <p>{percentage} Dice · {detail}</p>
            </article>
          ))}
        </div>

        <div className="evaluationContext">
          <section aria-labelledby="dataset-title">
            <div className="panelHeader"><div><span>01</span><h3 id="dataset-title">Evaluation dataset</h3></div><small>AUTHOR-DEFINED SPLIT</small></div>
            <dl>
              <div><dt>Source dataset</dt><dd>BraTS 2018</dd></div>
              <div><dt>Total studies</dt><dd>285</dd></div>
              <div><dt>Training split</dt><dd>200 studies</dd></div>
              <div><dt>Validation split</dt><dd>42 studies</dd></div>
              <div><dt>Held-out test split</dt><dd>43 studies</dd></div>
              <div><dt>Input per study</dt><dd>4 MRI volumes</dd></div>
            </dl>
          </section>

          <section aria-labelledby="speed-title">
            <div className="panelHeader"><div><span>02</span><h3 id="speed-title">Published speed benchmark</h3></div><small>A100 80 GB</small></div>
            <div className="speedTable" role="table" aria-label="Published reference timing in milliseconds">
              <div className="speedTableHead" role="row"><span role="columnheader">Method</span><span role="columnheader">Model</span><span role="columnheader">End-to-end</span></div>
              <div role="row"><strong role="cell">PyTorch FP32</strong><span role="cell">5.49 ms</span><span role="cell">592.01 ms</span></div>
              <div className="deployedMode" role="row"><strong role="cell">PyTorch AMP <i>configured</i></strong><span role="cell">4.36 ms</span><span role="cell">434.59 ms</span></div>
              <div role="row"><strong role="cell">TensorRT FP16 <i>not used</i></strong><span role="cell">2.09 ms</span><span role="cell">394.93 ms</span></div>
            </div>
            <p className="speedNote">These reference timings were measured on an NVIDIA A100 80 GB. NeuroLens AI runs on an NVIDIA L4, so its measured runtime is shown separately with each completed scan. Upload time, queueing, and GPU wake-up are not included here.</p>
          </section>
        </div>

      </section>

      <section className="boundary" id="boundaries">
        <p>ACADEMIC PROJECT / RESPONSIBLE USE</p>
        <h2>Review model-predicted tumor regions.<br /><em>Designed for learning, not diagnosis.</em></h2>
        <div><span>Use de-identified MRI data</span><span>Not for diagnosis</span><span>Review with medical experts</span></div>
      </section>

      <footer>
        <div className="brand"><span className="brandMark" aria-hidden="true"><i /><i /><i /></span><span>NeuroLens <b>AI</b></span></div>
        <p>Brain tumor segmentation across four MRI sequences</p>
        <p className="footerCredit">Developed by <a href="https://www.linkedin.com/in/smasifhossain" target="_blank" rel="noreferrer">S M Asif Hossain <i aria-hidden="true">in</i></a></p>
        <p><a href="https://opensource.org/license/mit" target="_blank" rel="noreferrer">MIT License</a> / 2026</p>
      </footer>
    </main>
  );
}
