import type React from "react"
import type { Metadata } from "next"
import { GeistSans } from "geist/font/sans"
import { GeistMono } from "geist/font/mono"
import { Analytics } from "@vercel/analytics/next"
import { Suspense } from "react"
import "./globals.css"

export const metadata: Metadata = {
  title: "AI Career Assistant",
  description:
    "Get personalized career guidance powered by advanced AI analysis. Connect your LinkedIn profile for tailored career insights and recommendations.",
  generator: "v0.app",
  keywords: ["AI", "career", "assistant", "LinkedIn", "career guidance", "professional development"],
  authors: [{ name: "AI Career Assistant" }],
  viewport: "width=device-width, initial-scale=1, maximum-scale=1, user-scalable=no, viewport-fit=cover",
  openGraph: {
    title: "AI Career Assistant",
    description: "Get personalized career guidance powered by advanced AI analysis",
    type: "website",
  },
}

export default function RootLayout({
  children,
}: Readonly<{
  children: React.ReactNode
}>) {
  return (
    <html lang="en">
      <body className={`font-sans ${GeistSans.variable} ${GeistMono.variable}`}>
        <Suspense fallback={<div>Loading...</div>}>{children}</Suspense>
        <Analytics />
      </body>
    </html>
  )
}
