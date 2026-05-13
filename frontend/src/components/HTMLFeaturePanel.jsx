// HTMLFeaturePanel.jsx
// แสดง HTML features ที่ดึงได้จากเว็บจริง
// แยก risk indicators (แดง) vs legitimacy indicators (เขียว)

const RISK_FEATURES = new Set([
  'has_login_form',
  'form_action_external',
  'meta_refresh',
  'num_iframes',
  'null_links_ratio',
  'num_hidden_elements',
])

const SAFE_FEATURES = new Set([
  'has_copyright',
  'has_favicon',
  'title_match_domain',
])

const FEATURE_LABELS = {
  num_external_links:   'External Links',
  num_internal_links:   'Internal Links',
  external_link_ratio:  'External Link Ratio',
  num_images:           'Images',
  num_scripts:          'Scripts',
  num_iframes:          'iFrames',
  has_favicon:          'Has Favicon',
  title_match_domain:   'Title Matches Domain',
  has_login_form:       'Has Login Form',
  form_action_external: 'Form Action External',
  null_links_ratio:     'Null Links Ratio',
  has_copyright:        'Has Copyright',
  meta_refresh:         'Meta Refresh',
  num_hidden_elements:  'Hidden Elements',
}

function FeatureRow({ name, value }) {
  const isRisk = RISK_FEATURES.has(name) && (value === 1 || value > 0)
  const isSafe = SAFE_FEATURES.has(name) && value === 1
  const label  = FEATURE_LABELS[name] ?? name

  // กำหนดสีตามความหมายของค่า
  let valueStyle = 'text-gray-300'
  let rowStyle   = ''
  let indicator  = null

  if (isRisk) {
    valueStyle = 'text-red-400 font-semibold'
    rowStyle   = 'bg-red-950/20'
    indicator  = <span className="text-red-500 text-xs ml-1">⚠</span>
  } else if (isSafe) {
    valueStyle = 'text-green-400 font-semibold'
    rowStyle   = 'bg-green-950/20'
    indicator  = <span className="text-green-500 text-xs ml-1">✓</span>
  }

  // แสดงค่า boolean เป็น Yes/No
  const displayValue = typeof value === 'number' && (value === 0 || value === 1)
    ? (value === 1 ? 'Yes' : 'No')
    : value

  return (
    <div className={`flex items-center justify-between px-3 py-1.5 rounded ${rowStyle}`}>
      <span className="text-gray-400 text-xs flex items-center gap-1">
        {label}
        {indicator}
      </span>
      <span className={`font-mono text-xs ${valueStyle}`}>
        {displayValue}
      </span>
    </div>
  )
}

function HTMLFeaturePanel({ features, status }) {
  if (!features || status !== 'ok') return null

  const entries = Object.entries(features)

  // นับ risk signals
  const riskCount = entries.filter(([k, v]) =>
    RISK_FEATURES.has(k) && (v === 1 || v > 0)
  ).length

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">

      {/* Header */}
      <div className="flex items-center justify-between">
        <h3 className="text-sm font-semibold text-gray-300">
          HTML Analysis
        </h3>
        <div className="flex items-center gap-2">
          {riskCount > 0 ? (
            <span className="text-xs bg-red-900/60 text-red-300 border border-red-800 px-2 py-0.5 rounded">
              {riskCount} risk signal{riskCount > 1 ? 's' : ''}
            </span>
          ) : (
            <span className="text-xs bg-green-900/60 text-green-300 border border-green-800 px-2 py-0.5 rounded">
              No risk signals
            </span>
          )}
        </div>
      </div>

      {/* Feature rows */}
      <div className="space-y-0.5">
        {entries.map(([name, value]) => (
          <FeatureRow key={name} name={name} value={value} />
        ))}
      </div>

      {/* Legend */}
      <div className="flex gap-4 text-xs text-gray-600 pt-1 border-t border-gray-800">
        <span className="flex items-center gap-1">
          <span className="text-red-500">⚠</span> Risk indicator
        </span>
        <span className="flex items-center gap-1">
          <span className="text-green-500">✓</span> Legitimacy indicator
        </span>
      </div>
    </div>
  )
}

export default HTMLFeaturePanel
