"use client"

import { useState } from "react"
import { Button } from "@/components/ui/button"
import { Card, CardContent, CardHeader } from "@/components/ui/card"
import { Loader2, Sparkles, MessageSquare, RotateCcw } from "lucide-react"
import { useRouter } from "next/navigation"

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false)
  const [loadingAction, setLoadingAction] = useState<"start" | "resume" | null>(null)
  const router = useRouter()

  const handleStartChat = async () => {
    setIsLoading(true)
    setLoadingAction("start")

    // Clear localStorage before starting new chat
    try {
      localStorage.removeItem("thread_id")
      localStorage.removeItem("chat_messages")
    } catch (e) {
      // ignore
    }

    await new Promise((resolve) => setTimeout(resolve, 1500))

    router.push("/start-chat")
    setIsLoading(false)
    setLoadingAction(null)
  }

  const handleResumeChat = async () => {
    setIsLoading(true)
    setLoadingAction("resume")

    await new Promise((resolve) => setTimeout(resolve, 1500))

    router.push("/resume-chat")
    setIsLoading(false)
    setLoadingAction(null)
  }

  return (
    <div className="min-h-screen bg-learntube-gradient flex items-center justify-center p-4">
      <div className="w-full max-w-md">
        <Card className="shadow-2xl border-0 bg-card">
          <CardHeader className="text-center space-y-6 pb-8">
            <div className="mx-auto w-20 h-20 bg-primary rounded-full flex items-center justify-center shadow-lg">
              <Sparkles className="h-10 w-10 text-white" />
            </div>
            <div className="space-y-3">
              <h1 className="text-3xl font-bold text-card-foreground">AI Career Assistant</h1>
              <p className="text-secondary-foreground text-lg leading-relaxed">
                Get personalized career guidance powered by advanced AI analysis
              </p>
            </div>
          </CardHeader>
          <CardContent className="space-y-4 pt-0">
            <Button
              onClick={handleStartChat}
              disabled={isLoading}
              className="w-full h-12 text-base font-semibold bg-primary hover:bg-primary/90 text-primary-foreground border-0 shadow-lg transition-all duration-200 transform hover:scale-[1.02]"
            >
              {isLoading && loadingAction === "start" ? (
                <>
                  <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                  Starting New Session...
                </>
              ) : (
                <>
                  <MessageSquare className="mr-3 h-5 w-5" />
                  Start New Chat
                </>
              )}
            </Button>

            <Button
              onClick={handleResumeChat}
              disabled={isLoading}
              variant="outline"
              className="w-full h-12 text-base font-semibold bg-card border-2 border-border hover:bg-card/80 text-card-foreground transition-all duration-200 transform hover:scale-[1.02]"
            >
              {isLoading && loadingAction === "resume" ? (
                <>
                  <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                  Loading Previous Chat...
                </>
              ) : (
                <>
                  <RotateCcw className="mr-3 h-5 w-5" />
                  Resume Previous Chat
                </>
              )}
            </Button>
          </CardContent>
        </Card>
      </div>
    </div>
  )
}
