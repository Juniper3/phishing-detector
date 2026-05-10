// SHAPChart.jsx
// Horizontal bar chart แสดง top 5 SHAP feature contributions
// สีแดง = increases_risk (ดัน model ไปทาง phishing)
// สีเขียว = decreases_risk (ดัน model ไปทาง legitimate)

import {
  BarChart, Bar, XAxis, YAxis,
  CartesianGrid, Tooltip, ReferenceLine,
  Cell, ResponsiveContainer,
} from 'recharts'

// Tooltip แบบ custom เพื่อให้เข้ากับ dark theme
function CustomTooltip({ active, payload }) {
  if (!active || !payload?.length) return null
  const d = payload[0].payload
  const isRisk = d.direction === 'increases_risk'

  return (
    <div className="bg-gray-800 border border-gray-700 rounded-lg px-3 py-2 text-xs shadow-lg">
      <p className="text-white font-medium mb-1 max-w-[200px] truncate">{d.name}</p>
      <p className="text-gray-400">
        ค่า feature:{' '}
        <span className="text-white font-mono">{d.value}</span>
      </p>
      <p className="text-gray-400">
        SHAP:{' '}
        <span className={`font-mono ${isRisk ? 'text-red-400' : 'text-green-400'}`}>
          {d.shap_value >= 0 ? '+' : ''}{d.shap_value.toFixed(4)}
        </span>
      </p>
      <p className={`mt-1 ${isRisk ? 'text-red-400' : 'text-green-400'}`}>
        {isRisk ? '↑ เพิ่มความเสี่ยง phishing' : '↓ ลดความเสี่ยง phishing'}
      </p>
    </div>
  )
}

function SHAPChart({ features }) {
  // แปลง features list ให้ recharts ใช้งานได้
  const data = features.map((f) => ({
    name:       f.name,
    value:      f.value,
    shap_value: f.shap_value,
    direction:  f.direction,
  }))

  return (
    <div className="bg-gray-900 border border-gray-800 rounded-xl p-5 space-y-4">

      {/* Header + legend */}
      <div className="flex items-center justify-between flex-wrap gap-2">
        <h3 className="text-sm font-semibold text-gray-300">
          SHAP Feature Contributions
        </h3>
        <div className="flex gap-4 text-xs text-gray-500">
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 bg-red-500 rounded-sm" />
            เพิ่มความเสี่ยง
          </span>
          <span className="flex items-center gap-1.5">
            <span className="inline-block w-2.5 h-2.5 bg-green-500 rounded-sm" />
            ลดความเสี่ยง
          </span>
        </div>
      </div>

      {/* Chart */}
      <ResponsiveContainer width="100%" height={240}>
        <BarChart
          data={data}
          layout="vertical"
          margin={{ top: 0, right: 24, left: 8, bottom: 0 }}
        >
          <CartesianGrid
            strokeDasharray="3 3"
            stroke="#1f2937"
            horizontal={false}
          />
          <XAxis
            type="number"
            tick={{ fill: '#6b7280', fontSize: 11 }}
            axisLine={{ stroke: '#374151' }}
            tickLine={false}
          />
          <YAxis
            type="category"
            dataKey="name"
            width={148}
            tick={{ fill: '#9ca3af', fontSize: 11 }}
            axisLine={false}
            tickLine={false}
          />
          <Tooltip
            content={<CustomTooltip />}
            cursor={{ fill: '#111827' }}
          />
          {/* เส้นอ้างอิงที่ 0 */}
          <ReferenceLine x={0} stroke="#374151" strokeWidth={1} />
          <Bar dataKey="shap_value" radius={[0, 3, 3, 0]}>
            {data.map((entry, i) => (
              <Cell
                key={i}
                fill={entry.direction === 'increases_risk' ? '#ef4444' : '#22c55e'}
              />
            ))}
          </Bar>
        </BarChart>
      </ResponsiveContainer>

      {/* คำอธิบาย */}
      <p className="text-gray-600 text-xs">
        SHAP value แสดงว่า feature แต่ละตัวมีผลต่อการตัดสินใจของ model มากแค่ไหน และในทิศทางใด
      </p>
    </div>
  )
}

export default SHAPChart
