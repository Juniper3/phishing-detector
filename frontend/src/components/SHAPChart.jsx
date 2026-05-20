import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
  Cell, ResponsiveContainer,
} from 'recharts'

function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const isRisk = d.direction === 'increases_risk'

  return (
    <div className="bg-white border border-slate-200 rounded-lg px-3 py-2.5 text-xs shadow-md">
      <p className="text-slate-800 font-medium mb-1.5 max-w-[200px] truncate">{d.name}</p>
      <p className="text-slate-500">
        ค่า feature:{' '}
        <span className="text-slate-700 font-mono font-medium">{d.value}</span>
      </p>
      <p className="text-slate-500">
        SHAP:{' '}
        <span className={`font-mono font-medium ${isRisk ? 'text-red-600' : 'text-green-600'}`}>
          {d.shap_value >= 0 ? '+' : ''}{d.shap_value.toFixed(4)}
        </span>
      </p>
      <p className={`mt-1.5 font-medium ${isRisk ? 'text-red-600' : 'text-green-600'}`}>
        {isRisk ? '↑ เพิ่มความเสี่ยง phishing' : '↓ ลดความเสี่ยง phishing'}
      </p>
    </div>
  )
}

function SHAPChart({ features }) {
  const data = features.map((f) => ({
    name:       f.name,
    value:      f.value,
    shap_value: f.shap_value,
    direction:  f.direction,
  }))

  return (
    <div className="bg-white border border-slate-200 rounded-xl p-5 space-y-4 shadow-sm">

      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-sm font-semibold text-slate-700">
          SHAP Feature Contributions
        </h3>
        <div className="flex gap-4 text-xs text-slate-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 bg-red-400 rounded-sm" />
            เพิ่มความเสี่ยง
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 bg-green-500 rounded-sm" />
            ลดความเสี่ยง
          </span>
        </div>
      </div>

      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#e2e8f0"
            horizontal={false}
          />
          <XAxis
            type="number"
            tick={{ fill: '#94a3b8', fontSize: 11 }}
            axisLine={{ stroke: '#cbd5e1' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={148}
            tick={{ fill: '#64748b', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: '#f8fafc' }}
          />
          <ReferenceLine x={0} stroke="#cbd5e1" strokeWidth={1} />
          <Bar dataKey="shap_value" radius={[0, 3, 3, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.direction === 'increases_risk' ? '#f87171' : '#4ade80'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      <p className="text-slate-400 text-xs">
        SHAP value แสดงว่า feature แต่ละตัวมีผลต่อการตัดสินใจของ model มากแค่ไหน และในทิศทางใด
      </p>
    </div>
  )
}

export default SHAPChart
