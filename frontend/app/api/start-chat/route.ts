import { type NextRequest, NextResponse } from "next/server"

const FASTAPI_BASE_URL = process.env.FASTAPI_BASE_URL || "http://127.0.0.1:8000"

export async function POST(request: NextRequest) {
  try {
    const body = await request.json()

    const response = await fetch(`${FASTAPI_BASE_URL}/start-chat`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(body),
    })

    if (!response.ok) {
      throw new Error(`FastAPI responded with status: ${response.status}`)
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error in start-chat:", error)
    return NextResponse.json({ error: "Failed to start chat session" }, { status: 500 })
  }
}
