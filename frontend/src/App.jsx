import { useState } from 'react'
import axios from 'axios'
import URLInput from './components/URLInput'
import ResultCard from './components/ResultCard'
import SHAPChart from './components/SHAPChart'
import HTMLFeaturePanel from './components/HTMLFeaturePanel'
import FeatureTable from './components/FeatureTable'
import URLHistory, { useURLHistory } from './components/URLHistory'

const API_URL = import.meta.env.VITE_API_URL ?? 'http://localhost:8001/api/v1/predict'

function App() {
  const [loading, setLoading]           = useState(false)
  const [result, setResult]             = useState(null)
  const [error, setError]               = useState(null)
  const [fetchHtml, setFetchHtml]       = useState(false)
  const [showFeatures, setShowFeatures] = useState(false)
  const [showHistory, setShowHistory]   = useState(false)

  const { history, addEntry, removeEntry, clearAll } = useURLHistory()

  const handleDetect = async (url) => {
    setLoading(true)
    setError(null)
    setResult(null)

    try {
      const { data } = await axios.post(API_URL, {
        url,
        fetch_html: fetchHtml,
      })
      setResult(data)
      addEntry(data)
      setShowHistory(true)
    } catch (err) {
      if (err.response) {
        setError(err.response.data?.detail ?? 'เกิดข้อผิดพลาดจาก server')
      } else {
        setError(`ไม่สามารถเชื่อมต่อ server ได้ — ตรวจสอบว่า backend รันอยู่ที่ ${API_URL}`)
      }
    } finally {
      setLoading(false)
    }
  }

  const handleRecheck = (url) => {
    handleDetect(url)
    window.scrollTo({ top: 0, behavior: 'smooth' })
  }

  return (
    <div className="min-h-screen bg-slate-50">

      {/* ─── Header ─── */}
      <header className="bg-white border-b border-slate-200 sticky top-0 z-10">
        <div className="max-w-2xl mx-auto px-6 py-4 flex items-center justify-between">
          <div className="flex items-center gap-3">
            <div className="w-8 h-8 bg-red-100 rounded-lg flex items-center justify-center flex-shrink-0">
              <span className="text-red-600 text-sm select-none">🛡</span>
            </div>
            <div>
              <h1 className="text-base font-semibold text-slate-800">Phishing Detector</h1>
              <p className="text-slate-400 text-xs">XGBoost + SHAP · IS DPU Research</p>
            </div>
          </div>

          {/* HTML Analysis Toggle */}
          <label className="flex items-center gap-2 cursor-pointer select-none">
            <span className="text-xs text-slate-500 font-medium">HTML</span>
            <div
              onClick={() => setFetchHtml(v => !v)}
              className={`relative w-9 h-5 rounded-full transition-colors duration-200 cursor-pointer
                ${fetchHtml ? 'bg-blue-500' : 'bg-slate-300'}`}
            >
              <div className={`absolute top-0.5 w-4 h-4 rounded-full bg-white shadow-sm
                transition-all duration-200 ${fetchHtml ? 'left-4' : 'left-0.5'}`}
              />
            </div>
          </label>
        </div>
      </header>

      {/* ─── Main ─── */}
      <main className="max-w-2xl mx-auto px-6 py-8 space-y-5">

        <p className="text-slate-500 text-sm">
          วิเคราะห์ URL ว่าเป็น phishing หรือ legitimate
          ด้วย machine learning พร้อม SHAP explanation
          {fetchHtml && (
            <span className="ml-2 text-blue-500 text-xs font-medium">· HTML mode เปิดอยู่</span>
          )}
        </p>

        <URLInput onDetect={handleDetect} loading={loading} />

        {loading && (
          <div className="flex items-center gap-2.5 text-slate-500 text-sm">
            <div className="w-4 h-4 border-2 border-slate-300 border-t-blue-500
              rounded-full animate-spin" />
            {fetchHtml ? 'กำลังวิเคราะห์ URL และดึง HTML...' : 'กำลังวิเคราะห์ URL...'}
          </div>
        )}

        {error && (
          <div className="bg-red-50 border border-red-200 text-red-700
            rounded-lg px-4 py-3 text-sm">
            {error}
          </div>
        )}

        {result && (
          <div className="space-y-4">
            <ResultCard result={result} />
            <SHAPChart features={result.top_features} />

            {/* Feature Table — collapsible */}
            <div>
              <button
                onClick={() => setShowFeatures(v => !v)}
                className="w-full flex items-center justify-between px-4 py-3
                  bg-white border border-slate-200 rounded-xl
                  hover:bg-slate-50 transition-colors"
              >
                <span className="text-sm text-slate-700 font-medium">
                  URL Feature Extraction
                </span>
                <span className="text-xs text-slate-400">
                  {showFeatures ? '▲ ซ่อน' : '▼ แสดง 9 features'}
                </span>
              </button>
              {showFeatures && (
                <div className="mt-1">
                  <FeatureTable
                    features={result.extracted_features}
                    topFeatures={result.top_features}
                  />
                </div>
              )}
            </div>

            {/* HTML Features */}
            {result.html_features && result.html_status === 'ok' && (
              <HTMLFeaturePanel
                features={result.html_features}
                status={result.html_status}
              />
            )}
            {result.html_status === 'timeout' && (
              <div className="bg-amber-50 border border-amber-200
                text-amber-700 rounded-lg px-4 py-3 text-xs">
                ⏱ HTML analysis timeout — ใช้ URL features เท่านั้น
              </div>
            )}
            {result.html_status === 'error' && (
              <div className="bg-slate-100 border border-slate-200
                text-slate-500 rounded-lg px-4 py-3 text-xs">
                ℹ HTML ดึงไม่ได้ — ใช้ URL features เท่านั้น
              </div>
            )}
          </div>
        )}

        {/* URL History */}
        {(history.length > 0 || result) && (
          <URLHistory
            history={history}
            onRemove={removeEntry}
            onClear={clearAll}
            onRecheck={handleRecheck}
            isOpen={showHistory}
            onToggle={() => setShowHistory(v => !v)}
          />
        )}

      </main>
    </div>
  )
}

export default App
