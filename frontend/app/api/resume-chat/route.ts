import { type NextRequest } from "next/server"

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://localhost:8000"

export async function GET(request: NextRequest) {
  try {
    const searchParams = request.nextUrl.searchParams
    const thread_id = searchParams.get('thread_id')
    const message = searchParams.get('message')

    if (!thread_id || !message) {
      return new Response('Missing thread_id or message', { status: 400 })
    }

    // Forward the streaming request to FastAPI
    const response = await fetch(
      `${FASTAPI_BASE_URL}/resume-chat?thread_id=${encodeURIComponent(thread_id)}&message=${encodeURIComponent(message)}`,
      {
        method: "GET",
        headers: {
          "Accept": "text/event-stream",
        },
      }
    )

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`)
    }

    // Return the streaming response directly
    return new Response(response.body, {
      headers: {
        'Content-Type': 'text/event-stream',
        'Cache-Control': 'no-cache',
        'Connection': 'keep-alive',
      },
    })
  } catch (error) {
    console.error("Error in resume-chat:", error)
    return new Response('Failed to resume chat', { status: 500 })
  }
}