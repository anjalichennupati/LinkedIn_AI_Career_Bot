"use client"

import { useState, useRef, useEffect } from "react"
import { Button } from "@/components/ui/button"
import { Textarea } from "@/components/ui/textarea"
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { Badge } from "@/components/ui/badge"
import { ScrollArea } from "@/components/ui/scroll-area"
import { Loader2, Send, User, Bot, Activity, LogOut, Sparkles, ArrowLeft } from "lucide-react"
import ReactMarkdown from "react-markdown"
import { useRouter } from "next/navigation"

interface Message {
  id: string
  content: string
  role: "user" | "assistant"
  timestamp: Date
}

interface MarketInsight {
  title: string
  url: string
}

export default function StartChatPage() {
  const [isInitializing, setIsInitializing] = useState(true)
  const [currentView, setCurrentView] = useState<"input" | "loading" | "chat">("input")
  const [linkedinUrl, setLinkedinUrl] = useState("")
  const [messages, setMessages] = useState<Message[]>([])
  const [currentMessage, setCurrentMessage] = useState("")
  const [isLoading, setIsLoading] = useState(false)
  const [isTyping, setIsTyping] = useState(false)
  const [threadId, setThreadId] = useState<string | null>(null)
  const [suggestedActions, setSuggestedActions] = useState<string[]>([])
  const [healthStatus, setHealthStatus] = useState<"healthy" | "error" | "checking">("healthy")
  const [profileName, setProfileName] = useState<string | null>(null)
  const [pendingAction, setPendingAction] = useState<string | null>(null)
  const messagesEndRef = useRef<HTMLDivElement>(null)
  const router = useRouter()

  // On mount, restore session from localStorage if exists
  useEffect(() => {
    try {
      const storedThreadId = localStorage.getItem("thread_id")
      const storedMessages = localStorage.getItem("chat_messages")
      
      if (storedThreadId && storedMessages) {
        // Restore the session
        setThreadId(storedThreadId)
        const parsed = JSON.parse(storedMessages)
        const messagesWithDates = parsed.map((msg: any) => ({
          ...msg,
          timestamp: new Date(msg.timestamp)
        }))
        setMessages(messagesWithDates)
        setCurrentView("chat")
      }
    } catch (e) {
      // If restore fails, stay on input view
    } finally {
      setIsInitializing(false)
    }
  }, [])

  // Save messages to localStorage whenever they change
  useEffect(() => {
    if (messages.length > 0) {
      try {
        localStorage.setItem("chat_messages", JSON.stringify(messages))
      } catch (e) {
        // ignore storage errors
      }
    }
  }, [messages])

  useEffect(() => {
    // Only scroll to bottom if user is near the bottom to prevent jarring scrolling
    const main = document.querySelector('main')
    if (main) {
      const isNearBottom = main.scrollTop + main.clientHeight >= main.scrollHeight - 100
      if (isNearBottom) {
        messagesEndRef.current?.scrollIntoView({ behavior: "smooth" })
      }
    }
  }, [messages, isTyping])

  // Fetch actual profile name from database
  useEffect(() => {
    const fetchProfileName = async () => {
      if (threadId) {
        try {
          const response = await fetch(`/api/profile/${threadId}`)
          if (response.ok) {
            const profileData = await response.json()
            if (profileData.profile_data && profileData.profile_data.name) {
              setProfileName(profileData.profile_data.name)
            }
          }
        } catch (error) {
          console.error("Failed to fetch profile name:", error)
        }
      }
    }
    
    fetchProfileName()
  }, [threadId])

  const startSession = async () => {
    if (!linkedinUrl.trim()) return

    setCurrentView("loading")
    setIsLoading(true)

    try {
      const response = await fetch("/api/start-chat", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ linkedin_url: linkedinUrl }),
      })

      if (response.ok) {
        const data = await response.json()
        setThreadId(data.thread_id)
        try {
          localStorage.setItem("thread_id", data.thread_id)
        } catch (e) {
          // ignore storage errors
        }

        const firstMessage: Message = {
          id: Date.now().toString(),
          content: "Hi! I'm from Learntube.ai. Your profile analysis is complete. Please select an option below or type a message to begin.",
          role: "assistant",
          timestamp: new Date(),
        }
        setMessages([firstMessage])
        setSuggestedActions([
          "enhance my full profile and give me the flaws in it",
          "Suggest best role with career plan.",
          "do a job fit analysis",
        ])
        setCurrentView("chat")
      }
    } catch (error) {
      console.error("Failed to start session:", error)
      setCurrentView("input")
    } finally {
      setIsLoading(false)
    }
  }

  const submitMessage = async (message: string) => {
    if (!message.trim() || !threadId) return
  
    const userMessage: Message = {
      id: Date.now().toString(),
      content: message,
      role: "user",
      timestamp: new Date(),
    }
  
    setMessages((prev) => [...prev, userMessage])
    setSuggestedActions([])
    setIsTyping(true)
  
    // Placeholder for streaming AI response
    const aiId = `${Date.now()}-ai`  // ← Keep this

  
    try {
      const response = await fetch(
        `/api/resume-chat?thread_id=${encodeURIComponent(threadId)}&message=${encodeURIComponent(message)}`,
        {
          method: "GET",
          headers: { "Accept": "text/event-stream" },
        }
      )
  
      const reader = response.body?.getReader()
      const decoder = new TextDecoder()
      let buffer = ""
  
      while (true) {
        const { done, value } = await reader!.read()
        if (done) break
  
        buffer += decoder.decode(value, { stream: true })
        const lines = buffer.split("\n\n")
        buffer = lines.pop() || ""
  
        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const data = JSON.parse(line.slice(6))
            
            if (data.type === 'start') {
              setIsTyping(false)
              // Create AI message NOW
              setMessages((prev) => [...prev, {
                id: aiId,
                content: "",
                role: "assistant",
                timestamp: new Date(),
                isStreaming: true,
              }])
            } else if (data.content) {
              setMessages((prev) =>
                prev.map((m) => m.id === aiId ? { ...m, content: data.content } : m)
              )
            } else if (data.done) {
              // Streaming complete
              break
            }
          }
        }
      }
    } catch (error) {
      console.error("Stream error:", error)
    } finally {
      setIsTyping(false)
    }
  }


  const sendMessage = () => {
    if (pendingAction) {
      const combinedMessage = `${pendingAction}: ${currentMessage}`
      submitMessage(combinedMessage)
      setPendingAction(null)
    } else {
      submitMessage(currentMessage)
    }
    setCurrentMessage("")
  }

  const handleSuggestedActionClick = (action: string) => {
    const multiStepActions: { [key: string]: string } = {
      "do a job fit analysis":
        "Certainly. Please provide the job description you'd like me to analyze.",
    }

    if (multiStepActions[action]) {
      setPendingAction(action)
      const followUpMessage: Message = {
        id: Date.now().toString(),
        content: multiStepActions[action],
        role: "assistant",
        timestamp: new Date(),
      }
      setMessages((prev) => [...prev, followUpMessage])
      setSuggestedActions([])
    } else {
      submitMessage(action)
    }
  }

  const endSession = async () => {
    if (!threadId) return

    try {
      await fetch(`/api/chat/${threadId}`, {
        method: "DELETE",
      })
    } catch (error) {
      console.error("Failed to end session:", error)
    } finally {
      try {
        localStorage.removeItem("thread_id")
        localStorage.removeItem("chat_messages")
      } catch (e) {
        // ignore storage errors
      }
      setThreadId(null)
      setMessages([])
      setLinkedinUrl("")
      setCurrentView("input")
    }
  }

  const checkHealth = async () => {
    setHealthStatus("checking")
    try {
      const response = await fetch("/api/health")
      setHealthStatus(response.ok ? "healthy" : "error")
    } catch (error) {
      setHealthStatus("error")
    }
  }

  const extractMarketInsights = (content: string): { content: string; insights: MarketInsight[] } => {
    const insightsRegex = /### Market Insights\s*([\s\S]*?)(?=\n###|\n\n|$)/
    const match = content.match(insightsRegex)

    if (!match) return { content, insights: [] }

    const insightsSection = match[1]
    const linkRegex = /\[([^\]]+)\]$$([^)]+)$$/g
    const insights: MarketInsight[] = []
    let linkMatch

    while ((linkMatch = linkRegex.exec(insightsSection)) !== null) {
      insights.push({
        title: linkMatch[1],
        url: linkMatch[2],
      })
    }

    const contentWithoutInsights = content.replace(insightsRegex, "").trim()
    return { content: contentWithoutInsights, insights }
  }

  const TypingIndicator = () => (
    <div className="flex items-start space-x-2 max-w-[85%] mb-4">
      <div className="w-7 h-7 rounded-full bg-muted text-muted-foreground flex items-center justify-center">
        <Bot className="h-3 w-3" />
      </div>
      <div className="bg-muted rounded-2xl px-4 py-2 flex items-center space-x-2">
        <div className="flex space-x-1">
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
          <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
        </div>
        <span className="text-sm text-foreground">AI is typing...</span>
      </div>
    </div>
  )

  // Show nothing while checking localStorage to prevent flash
  if (isInitializing) {
    return null
  }

  if (currentView === "loading") {
    return (
      <div className="min-h-screen bg-learntube-gradient flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <Card className="shadow-2xl border-0 bg-card">
            <CardContent className="p-8 text-center space-y-6">
              <div className="mx-auto w-20 h-20 bg-primary rounded-full flex items-center justify-center shadow-lg">
                <Loader2 className="h-10 w-10 text-white animate-spin" />
              </div>
              <div className="space-y-3">
                <h2 className="text-2xl font-bold text-card-foreground">Analyzing Your Profile</h2>
                <p className="text-secondary-foreground text-base leading-relaxed">
                  We're scraping your LinkedIn data and preparing your personalized career insights...
                </p>
              </div>
              <div className="space-y-4">
                <div className="flex items-center justify-center space-x-2 text-sm text-muted-foreground">
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "0ms" }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "150ms" }} />
                  <div className="w-2 h-2 bg-primary rounded-full animate-bounce" style={{ animationDelay: "300ms" }} />
                  <span className="ml-2">Processing your data</span>
                </div>
                <div className="text-xs text-muted-foreground">
                  This may take a few moments while we gather your professional information
                </div>
              </div>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  if (currentView === "input") {
    return (
      <div className="min-h-screen bg-learntube-gradient flex items-center justify-center p-4">
        <div className="w-full max-w-md">
          <Card className="shadow-2xl border-0 bg-card">
            <CardHeader className="text-center space-y-6 pb-8">
              <Button
                variant="ghost"
                onClick={() => router.push("/")}
                className="absolute top-4 left-4 text-card-foreground hover:bg-card/80"
              >
                <ArrowLeft className="h-4 w-4 mr-2" />
                Back
              </Button>
              <div className="mx-auto w-20 h-20 bg-primary rounded-full flex items-center justify-center shadow-lg">
                <Sparkles className="h-10 w-10 text-white" />
              </div>
              <div className="space-y-3">
                <h1 className="text-3xl font-bold text-card-foreground">Start New Chat</h1>
                <p className="text-secondary-foreground text-lg leading-relaxed">
                  Enter your LinkedIn profile to begin career analysis
                </p>
              </div>
            </CardHeader>
            <CardContent className="space-y-6 pt-0">
              <div className="space-y-3">
                <label htmlFor="linkedin" className="text-sm font-semibold text-card-foreground block">
                  LinkedIn Profile URL
                </label>
                <input
                  id="linkedin"
                  placeholder="https://linkedin.com/in/yourprofile"
                  value={linkedinUrl}
                  onChange={(e: React.ChangeEvent<HTMLInputElement>) => setLinkedinUrl(e.target.value)}
                  onKeyDown={(e: React.KeyboardEvent<HTMLInputElement>) => e.key === "Enter" && startSession()}
                  className="h-12 text-base bg-input border-2 border-border focus:border-primary transition-colors text-card-foreground placeholder:text-muted-foreground w-full rounded-md px-3"
                />
              </div>
              <Button
                onClick={startSession}
                disabled={!linkedinUrl.trim() || isLoading}
                className="w-full h-12 text-base font-semibold bg-primary hover:bg-primary/90 text-primary-foreground border-0 shadow-lg transition-all duration-200 transform hover:scale-[1.02]"
              >
                {isLoading ? (
                  <>
                    <Loader2 className="mr-3 h-5 w-5 animate-spin" />
                    Analyzing Your Profile...
                  </>
                ) : (
                  <>
                    <Sparkles className="mr-3 h-5 w-5" />
                    Start Career Analysis
                  </>
                )}
              </Button>
            </CardContent>
          </Card>
        </div>
      </div>
    )
  }

  return (
    <div className="flex flex-col h-screen bg-card text-card-foreground overflow-hidden">
      {/* Mobile-Optimized Header */}
      <header className="sticky top-0 z-10 px-3 py-2 border-b bg-card/95 backdrop-blur-sm border-border mobile-ui">
        <div className="flex items-center justify-between w-full">
          {/* Left side - Profile info */}
          <div className="flex items-center space-x-2 flex-1 min-w-0">
            <div className="w-8 h-8 bg-primary rounded-full flex items-center justify-center flex-shrink-0">
              <Sparkles className="h-4 w-4 text-white" />
            </div>
            <div className="min-w-0 flex-1">
              <h1 className="text-sm font-bold text-foreground truncate">
                {profileName || "AI Career Assistant"}
              </h1>
              <p className="text-xs text-muted-foreground">Personalized career guidance</p>
            </div>
          </div>

          {/* Right side - Action buttons */}
          <div className="flex items-center space-x-1 flex-shrink-0">
            <Button
              variant="outline"
              size="sm"
              onClick={checkHealth}
              disabled={healthStatus === "checking"}
              className="bg-card border-border text-card-foreground hover:bg-card/80 h-8 px-2"
            >
              {healthStatus === "checking" ? (
                <Loader2 className="h-3 w-3 animate-spin" />
              ) : (
                <Activity className="h-3 w-3" />
              )}
              <Badge
                variant={healthStatus === "healthy" ? "default" : "destructive"}
                className="ml-1 bg-primary text-primary-foreground text-xs px-1 py-0"
              >
                {healthStatus === "checking" ? "..." : healthStatus}
              </Badge>
            </Button>

            <Button
              variant="outline"
              size="sm"
              onClick={endSession}
              className="bg-card border-border text-card-foreground hover:bg-destructive/20 hover:border-destructive h-8 px-2"
            >
              <LogOut className="h-3 w-3" />
              <span className="ml-1 text-xs hidden sm:inline">End</span>
            </Button>
          </div>
        </div>
      </header>

      {/* Chat Messages */}
      <main className="flex-1 overflow-y-auto bg-learntube-gradient relative">
        <div className="max-w-4xl px-3 py-4 mx-auto">
          <div className="space-y-3 pb-4">
            {messages.map((message) => {
              const { content, insights } =
                message.role === "assistant"
                  ? extractMarketInsights(message.content)
                  : { content: message.content, insights: [] }

              return (
                <div key={message.id}>
                  <div className={`flex ${message.role === "user" ? "justify-end" : "justify-start"} mb-4`}>
                    <div
                      className={`flex items-start space-x-2 max-w-[85%] ${
                        message.role === "user" ? "flex-row-reverse space-x-reverse" : ""
                      }`}
                    >
                      <div
                        className={`w-7 h-7 rounded-full flex-shrink-0 flex items-center justify-center ${
                          message.role === "user"
                            ? "bg-primary text-primary-foreground"
                            : "bg-muted text-muted-foreground"
                        }`}
                      >
                        {message.role === "user" ? <User className="h-3 w-3" /> : <Bot className="h-3 w-3" />}
                      </div>
                      <div
                        className={`rounded-2xl px-4 py-2 shadow-sm break-words max-w-full overflow-hidden ${
                          message.role === "user"
                            ? "bg-user-message text-white"
                            : "bg-muted text-foreground"
                        }`}
                      >
                        <div className="text-sm leading-relaxed max-w-full overflow-wrap-anywhere">
                          {message.role === "user" ? (
                            <p className="mb-0 leading-relaxed text-sm">{content}</p>
                          ) : (
                            <div className="prose prose-sm max-w-none [&>*:first-child]:mt-0 [&>*:last-child]:mb-0">
                              <ReactMarkdown
                                components={{
                                  h1: ({ children }) => (
                                    <h1 className="text-base font-bold mb-2 text-foreground">{children}</h1>
                                  ),
                                  h2: ({ children }) => (
                                    <h2 className="text-sm font-semibold mb-1 text-foreground">{children}</h2>
                                  ),
                                  h3: ({ children }) => (
                                    <h3 className="text-sm font-semibold mb-1 text-foreground">{children}</h3>
                                  ),
                                  p: ({ children }) => (
                                    <p className="mb-1 last:mb-0 leading-relaxed text-sm text-foreground">{children}</p>
                                  ),
                                  ul: ({ children }) => <ul className="mb-1 last:mb-0 pl-4 space-y-1">{children}</ul>,
                                  li: ({ children }) => <li className="text-foreground leading-relaxed text-sm">{children}</li>,
                                  a: ({ children, href }) => (
                                    <a
                                      href={href}
                                      className="text-primary hover:underline text-sm"
                                      target="_blank"
                                      rel="noopener noreferrer"
                                    >
                                      {children}
                                    </a>
                                  ),
                                  strong: ({ children }) => (
                                    <strong className="font-semibold text-foreground">{children}</strong>
                                  ),
                                }}
                              >
                                {content}
                              </ReactMarkdown>
                            </div>
                          )}
                        </div>
                        {message.content && (
  <div
    className={`text-xs mt-1 ${
      message.role === "user" ? "text-white/70" : "text-muted-foreground"
    }`}
  >
    {message.timestamp.toLocaleTimeString([], { hour: '2-digit', minute: '2-digit' })}
  </div>
)}
                      </div>
                    </div>
                  </div>

                  {insights.length > 0 && (
                    <div className="mt-4 ml-11">
                      <Card className="bg-card border-border">
                        <CardHeader className="pb-3">
                          <CardTitle className="text-sm font-semibold text-card-foreground flex items-center">
                            <Activity className="h-4 w-4 mr-2 text-primary" />
                            Market Insights
                          </CardTitle>
                        </CardHeader>
                        <CardContent className="pt-0">
                          <div className="space-y-2">
                            {insights.map((insight, index) => (
                              <a
                                key={index}
                                href={insight.url}
                                target="_blank"
                                rel="noopener noreferrer"
                                className="block p-2 rounded-md bg-muted hover:bg-muted/80 transition-colors border border-border hover:border-primary/30"
                              >
                                <span className="text-sm font-medium text-primary hover:underline">
                                  {insight.title}
                                </span>
                              </a>
                            ))}
                          </div>
                        </CardContent>
                      </Card>
                    </div>
                  )}
                </div>
              )
            })}

            {suggestedActions.length > 0 && (
              <div className="flex justify-start mb-4">
                <div className="ml-9 flex flex-col items-start gap-2">
                  {suggestedActions.map((action, index) => (
                    <Button
                      key={index}
                      variant="outline"
                      size="sm"
                      onClick={() => handleSuggestedActionClick(action)}
                      className="bg-muted hover:bg-muted/80 border-muted text-foreground h-auto py-2 px-3 text-sm rounded-xl"
                    >
                      <span className="font-normal">{action}</span>
                    </Button>
                  ))}
                </div>
              </div>
            )}

            {isTyping && <TypingIndicator />}
            <div ref={messagesEndRef} />
          </div>
        </div>
      </main>

      {/* Message Input */}
      <footer className="sticky bottom-0 p-1 bg-card/95 backdrop-blur-sm border-t border-border mobile-ui">
        <div className="max-w-4xl mx-auto">
          <div className="flex items-end space-x-2 bg-muted rounded-2xl p-1">
            <Textarea
              placeholder="Explore roles, skills, or growth..."
              value={currentMessage}
              onChange={(e) => {
                setCurrentMessage(e.target.value)
                // Auto-resize
                const target = e.target as HTMLTextAreaElement
                target.style.height = "auto"
                target.style.height = `${Math.min(target.scrollHeight, 120)}px`
              }}
              onKeyDown={(e) => {
                if (e.key === "Enter" && !e.shiftKey) {
                  e.preventDefault()
                  sendMessage()
                  // Reset height after sending
                  setTimeout(() => {
                    const target = e.target as HTMLTextAreaElement
                    target.style.height = "auto"
                  }, 0)
                }
              }}
              disabled={isTyping}
              className="flex-1 bg-transparent border-0 focus-visible:ring-0 focus-visible:ring-offset-0 text-foreground placeholder:text-muted-foreground resize-none text-sm leading-relaxed p-2 outline-none"
              style={{ minHeight: "20px", maxHeight: "120px" }}
              rows={1}
            />
            <Button
              onClick={() => {
                sendMessage()
                // Reset height after sending
                setTimeout(() => {
                  const textarea = document.querySelector('textarea') as HTMLTextAreaElement
                  if (textarea) {
                    textarea.style.height = "auto"
                  }
                }, 0)
              }}
              disabled={!currentMessage.trim() || isTyping}
              className="h-8 w-8 p-0 rounded-full bg-primary hover:bg-primary/90 text-primary-foreground shadow-sm flex-shrink-0"
            >
              <Send className="w-3 h-3" />
              <span className="sr-only">Send Message</span>
            </Button>
          </div>
        </div>
      </footer>
    </div>
  )
}