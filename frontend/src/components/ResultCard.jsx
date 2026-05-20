function RiskBar({ score }) {
  const barColor =
    score >= 60 ? 'bg-red-500'
    : score >= 30 ? 'bg-amber-400'
    : 'bg-green-500'

  const trackColor =
    score >= 60 ? 'bg-red-100'
    : score >= 30 ? 'bg-amber-100'
    : 'bg-green-100'

  return (
    <div className="space-y-1.5">
      <div className="flex justify-between text-xs text-slate-500">
        <span>Risk Score</span>
        <span className="font-mono font-medium text-slate-700">{score} / 100</span>
      </div>
      <div className={`h-2 ${trackColor} rounded-full overflow-hidden`}>
        <div
          className={`h-full rounded-full transition-all duration-700 ${barColor}`}
          style={{ width: `${score}%` }}
        />
      </div>
    </div>
  )
}

function ResultCard({ result }) {
  const isPhishing = result.prediction === 'phishing'

  const cardStyle = isPhishing
    ? 'bg-red-50 border-red-200'
    : 'bg-green-50 border-green-200'

  const badgeStyle = isPhishing
    ? 'bg-red-100 text-red-700 border border-red-300'
    : 'bg-green-100 text-green-700 border border-green-300'

  const labelText = isPhishing ? 'Phishing' : 'Legitimate'
  const icon      = isPhishing ? '⚠' : '✓'

  const confidencePct = (result.confidence * 100).toFixed(1)

  return (
    <div className={`border rounded-xl p-5 space-y-4 ${cardStyle}`}>

      <p className="text-slate-500 text-xs font-mono truncate" title={result.url}>
        {result.url}
      </p>

      <div className="flex items-center justify-between flex-wrap gap-3">
        <span className={`inline-flex items-center gap-1.5 text-sm font-semibold px-3 py-1.5 rounded-lg ${badgeStyle}`}>
          <span>{icon}</span>
          {labelText.toUpperCase()}
        </span>
        <div className="text-right">
          <span className="text-slate-400 text-xs block">Confidence</span>
          <span className={`font-bold text-2xl leading-none ${isPhishing ? 'text-red-600' : 'text-green-600'}`}>
            {confidencePct}%
          </span>
        </div>
      </div>

      <RiskBar score={result.risk_score} />

      <p className="text-slate-400 text-xs text-right font-mono">
        {result.processing_time_ms} ms
      </p>
    </div>
  )
}

export default ResultCard
