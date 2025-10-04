import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card"
import { ExternalLink, TrendingUp } from "lucide-react"

interface MarketInsight {
  title: string
  url: string
  description?: string
}

interface MarketInsightsProps {
  insights: MarketInsight[]
}

export function MarketInsights({ insights }: MarketInsightsProps) {
  if (!insights.length) return null

  return (
    <Card className="mt-4 border-l-4 border-l-primary">
      <CardHeader className="pb-3">
        <CardTitle className="flex items-center space-x-2 text-lg">
          <TrendingUp className="h-5 w-5 text-primary" />
          <span>Market Insights</span>
        </CardTitle>
      </CardHeader>
      <CardContent className="space-y-3">
        {insights.map((insight, index) => (
          <div key={index} className="flex items-start justify-between p-3 bg-muted/50 rounded-lg">
            <div className="flex-1">
              <h4 className="font-medium text-sm">{insight.title}</h4>
              {insight.description && <p className="text-xs text-muted-foreground mt-1">{insight.description}</p>}
            </div>
            <a
              href={insight.url}
              target="_blank"
              rel="noopener noreferrer"
              className="ml-2 p-1 hover:bg-background rounded transition-colors"
            >
              <ExternalLink className="h-4 w-4 text-muted-foreground hover:text-primary" />
            </a>
          </div>
        ))}
      </CardContent>
    </Card>
  )
}
