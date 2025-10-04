import { NextRequest, NextResponse } from "next/server"

export async function GET(request: NextRequest, { params }: { params: { thread_id: string } }) {
  const thread_id = params.thread_id
  const backendApiUrl = process.env.BACKEND_API_URL || "http://127.0.0.1:8000"

  try {
    const response = await fetch(`${backendApiUrl}/profile/${thread_id}`)

    if (!response.ok) {
      const errorData = await response.text()
      console.error(`Error from backend: ${response.status} ${response.statusText}`, errorData)
      return NextResponse.json({ error: "Failed to fetch profile data from backend" }, { status: response.status })
    }

    const data = await response.json()
    return NextResponse.json(data)
  } catch (error) {
    console.error("Error fetching profile data:", error)
    return NextResponse.json({ error: "Internal server error" }, { status: 500 })
  }
}
