// ResultCard.jsx
// แสดงผล prediction — สีแดง = Phishing, สีเขียว = Legitimate

// Risk score bar 0-100
function RiskBar({ score }) {
  // เกิน 60 = สูง (แดง), 30-60 = กลาง (เหลือง), ต่ำกว่า 30 = ต่ำ (เขียว)
  const barColor =
    score >= 60 ? 'bg-red-500'
    : score >= 30 ? 'bg-yellow-500'
    : 'bg-green-500'

  return (
    <div className="space-y-1">
      <div className="flex justify-between text-xs text-gray-400">
        <span>Risk Score</span>
        <span className="font-mono">{score} / 100</span>
      </div>
      <div className="h-1.5 bg-gray-800 rounded-full overflow-hidden">
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

  // สีตาม prediction
  const borderColor = isPhishing ? 'border-red-800'   : 'border-green-800'
  const badgeStyle  = isPhishing
    ? 'bg-red-900/60 text-red-300 border border-red-700'
    : 'bg-green-900/60 text-green-300 border border-green-700'
  const labelText   = isPhishing ? '⚠ PHISHING' : '✓ LEGITIMATE'

  const confidencePct = (result.confidence * 100).toFixed(1)

  return (
    <div className={`bg-gray-900 border ${borderColor} rounded-xl p-5 space-y-4`}>

      {/* URL ที่ตรวจสอบ */}
      <p className="text-gray-500 text-xs font-mono truncate" title={result.url}>
        {result.url}
      </p>

      {/* Prediction badge + Confidence */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <span className={`text-sm font-bold tracking-widest px-3 py-1 rounded ${badgeStyle}`}>
          {labelText}
        </span>
        <div className="text-right">
          <span className="text-gray-400 text-xs block">Confidence</span>
          <span className="text-white font-semibold text-lg leading-none">
            {confidencePct}%
          </span>
        </div>
      </div>

      {/* Risk score bar */}
      <RiskBar score={result.risk_score} />

      {/* Processing time */}
      <p className="text-gray-700 text-xs text-right font-mono">
        {result.processing_time_ms} ms
      </p>
    </div>
  )
}

export default ResultCard
