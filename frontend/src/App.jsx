import { useState } from 'react'
import axios from 'axios'
import URLInput from './components/URLInput'
import ResultCard from './components/ResultCard'
import SHAPChart from './components/SHAPChart'

// URL ของ backend API
const API_URL = 'http://localhost:8001/api/v1/predict'

function App() {
  const [loading, setLoading] = useState(false)
  const [result, setResult]   = useState(null)
  const [error, setError]     = useState(null)

  // ส่ง URL ไป backend แล้วเก็บผลลัพธ์
  const handleDetect = async (url) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const { data } = await axios.post(API_URL, { url })
      setResult(data)
    } catch (err) {
      if (err.response) {
        setError(err.response.data?.detail ?? 'เกิดข้อผิดพลาดจาก server')
      } else {
        setError('ไม่สามารถเชื่อมต่อ server ได้ — ตรวจสอบว่า backend รันอยู่ที่ port 8001')
      }
    } finally {
      setLoading(false)
    }
  }

  return (
    <div className="min-h-screen bg-gray-950 text-white">

      {/* ─── Header ─── */}
      <header className="border-b border-gray-800">
        <div className="max-w-3xl mx-auto px-4 py-4 flex items-center gap-3">
          <span className="text-red-500 text-xl select-none">⚠</span>
          <div>
            <h1 className="text-lg font-bold tracking-wide">Phishing Detector</h1>
            <p className="text-gray-500 text-xs">XGBoost + SHAP · IS DPU Research</p>
          </div>
        </div>
      </header>

      {/* ─── Main Content ─── */}
      <main className="max-w-3xl mx-auto px-4 py-10 space-y-6">

        <p className="text-gray-500 text-sm">
          วิเคราะห์ URL ว่าเป็น phishing หรือ legitimate
          ด้วย machine learning พร้อม SHAP explanation
        </p>

        {/* ช่องใส่ URL */}
        <URLInput onDetect={handleDetect} loading={loading} />

        {/* Error message */}
        {error && (
          <div className="bg-red-950/60 border border-red-800 text-red-300 rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {/* ผลลัพธ์ */}
        {result && (
          <div className="space-y-4">
            <ResultCard result={result} />
            <SHAPChart features={result.top_features} />
          </div>
        )}
      </main>
    </div>
  )
}

export default App
