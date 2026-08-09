export const dynamic = "force-dynamic";

export async function GET() {
  const workerUrl = process.env.NEUROLENS_WORKER_URL?.trim() ?? "";

  return Response.json(
    { workerUrl },
    { headers: { "cache-control": "no-store, max-age=0" } },
  );
}
