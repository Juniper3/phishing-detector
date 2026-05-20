const FEATURE_META = {
  url_length: {
    label: 'URL Length',
    desc: 'ความยาวรวมของ URL',
    unit: 'chars',
    risky: (v) => v > 75,
  },
  hostname_length: {
    label: 'Hostname Length',
    desc: 'ความยาวของ domain',
    unit: 'chars',
    risky: (v) => v > 30,
  },
  has_ip: {
    label: 'IP as Hostname',
    desc: 'ใช้ IP address แทน domain',
    unit: 'bool',
    risky: (v) => v === 1,
  },
  num_digits: {
    label: 'Digit Count',
    desc: 'จำนวนตัวเลขใน URL',
    unit: 'count',
    risky: (v) => v > 10,
  },
  digit_ratio: {
    label: 'Digit Ratio',
    desc: 'สัดส่วนตัวเลขต่อความยาว URL',
    unit: 'ratio',
    risky: (v) => v > 0.15,
  },
  special_char_ratio: {
    label: 'Special Char Ratio',
    desc: 'สัดส่วนอักขระพิเศษ',
    unit: 'ratio',
    risky: (v) => v > 0.2,
  },
  url_entropy: {
    label: 'URL Entropy',
    desc: 'ความสุ่มของ URL (Shannon entropy)',
    unit: 'bits',
    risky: (v) => v > 4.0,
  },
  num_subdomains: {
    label: 'Subdomain Count',
    desc: 'จำนวน subdomain',
    unit: 'count',
    risky: (v) => v > 2,
  },
  num_equal: {
    label: 'Equal Sign Count',
    desc: 'จำนวนเครื่องหมาย = ใน URL',
    unit: 'count',
    risky: (v) => v > 3,
  },
}

function FeatureRow({ name, value, shapInfo }) {
  const meta = FEATURE_META[name]
  if (!meta) return null

  const isRisky = meta.risky(value)
  const isBool  = meta.unit === 'bool'

  let display = value
  if (isBool)                    display = value === 1 ? 'Yes' : 'No'
  else if (meta.unit === 'ratio') display = value.toFixed(4)
  else if (meta.unit === 'bits')  display = value.toFixed(3)

  return (
    <tr className={`border-b border-slate-100 transition-colors hover:bg-slate-50
      ${isRisky ? 'bg-red-50/60' : ''}`}>

      <td className="py-2.5 px-4">
        <div className="flex items-center gap-2 flex-wrap">
          <span className="text-xs font-medium text-slate-700">{meta.label}</span>
          {shapInfo && (
            <span className={`text-xs font-mono px-1.5 py-0.5 rounded border leading-none
              ${shapInfo.direction === 'increases_risk'
                ? 'text-red-600 border-red-200 bg-red-50'
                : 'text-green-600 border-green-200 bg-green-50'}`}>
              SHAP {shapInfo.shap_value >= 0 ? '+' : ''}{shapInfo.shap_value.toFixed(3)}
            </span>
          )}
        </div>
        <div className="text-xs text-slate-400 mt-0.5">{meta.desc}</div>
      </td>

      <td className="py-2.5 px-4 text-right">
        <span className={`font-mono text-sm font-semibold
          ${isRisky ? 'text-red-600' : 'text-green-600'}`}>
          {display}
        </span>
        <span className="text-slate-400 text-xs ml-1">{meta.unit !== 'bool' ? meta.unit : ''}</span>
      </td>

      <td className="py-2.5 px-4 text-center w-12">
        {isRisky
          ? <span className="text-red-400 text-xs">⚠</span>
          : <span className="text-green-500 text-xs">✓</span>
        }
      </td>
    </tr>
  )
}

function FeatureTable({ features, topFeatures = [] }) {
  if (!features) return null

  const shapMap = Object.fromEntries(
    topFeatures.map(f => [f.name, { shap_value: f.shap_value, direction: f.direction }])
  )

  const riskCount = Object.entries(features).filter(([k, v]) =>
    FEATURE_META[k]?.risky(v)
  ).length

  const total     = Object.keys(FEATURE_META).length
  const shapCount = topFeatures.length

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">

      <div className="flex items-center justify-between px-4 py-3 border-b border-slate-100">
        <div>
          <h3 className="text-sm font-semibold text-slate-700">URL Feature Extraction</h3>
          <p className="text-xs text-slate-400 mt-0.5">
            ค่า features ที่ model ใช้ตัดสินใจ ({total} features
            {shapCount > 0 && ` · ${shapCount} ใน SHAP top`})
          </p>
        </div>
        <div className={`text-xs px-2.5 py-1 rounded-lg border font-mono font-medium
          ${riskCount > 0
            ? 'bg-red-50 text-red-600 border-red-200'
            : 'bg-green-50 text-green-600 border-green-200'}`}>
          {riskCount} / {total} risky
        </div>
      </div>

      <table className="w-full">
        <thead>
          <tr className="border-b border-slate-100 bg-slate-50">
            <th className="py-2 px-4 text-left text-xs text-slate-500 font-medium">Feature</th>
            <th className="py-2 px-4 text-right text-xs text-slate-500 font-medium">Value</th>
            <th className="py-2 px-4 text-center text-xs text-slate-500 font-medium w-12">Risk</th>
          </tr>
        </thead>
        <tbody>
          {Object.keys(FEATURE_META).map((name) => (
            <FeatureRow
              key={name}
              name={name}
              value={features[name] ?? 0}
              shapInfo={shapMap[name] ?? null}
            />
          ))}
        </tbody>
      </table>

      <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50 flex items-center justify-between">
        <p className="text-xs text-slate-400">
          ⚠ = ค่าเกิน threshold · ✓ = ค่าปกติ · SHAP = ผลต่อการตัดสินใจ
        </p>
        <p className="text-xs text-slate-400 font-mono">
          XGBoost · threshold=0.437
        </p>
      </div>
    </div>
  )
}

export default FeatureTable
