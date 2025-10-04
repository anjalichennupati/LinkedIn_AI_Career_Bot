import { type NextRequest, NextResponse } from "next/server"

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://localhost:8000"

export async function DELETE(request: NextRequest, { params }: { params: { thread_id: string } }) {
  try {
    const { thread_id } = params

    const response = await fetch(`${FASTAPI_BASE_URL}/chat/${thread_id}`, {
      method: "DELETE",
      headers: {
        "Content-Type": "application/json",
      },
    })

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error in delete-chat:", error)
    return NextResponse.json({ error: "Failed to delete chat session" }, { status: 500 })
  }
}
