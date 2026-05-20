import { useState, useEffect } from 'react'

const STORAGE_KEY = 'phishing_detector_history'
const MAX_HISTORY = 20

export function useURLHistory() {
  const [history, setHistory] = useState([])

  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) setHistory(JSON.parse(stored))
    } catch {
      setHistory([])
    }
  }, [])

  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    } catch {}
  }, [history])

  const addEntry = (result) => {
    const entry = {
      id:         Date.now(),
      url:        result.url,
      prediction: result.prediction,
      confidence: result.confidence,
      risk_score: result.risk_score,
      timestamp:  new Date().toISOString(),
    }
    setHistory(prev => [entry, ...prev].slice(0, MAX_HISTORY))
  }

  const removeEntry = (id) => {
    setHistory(prev => prev.filter(e => e.id !== id))
  }

  const clearAll = () => setHistory([])

  return { history, addEntry, removeEntry, clearAll }
}

function HistoryItem({ entry, onRemove, onRecheck }) {
  const isPhishing = entry.prediction === 'phishing'
  const date = new Date(entry.timestamp)
  const timeStr = date.toLocaleTimeString('th-TH', {
    hour: '2-digit', minute: '2-digit',
  })
  const dateStr = date.toLocaleDateString('th-TH', {
    day: '2-digit', month: 'short',
  })

  return (
    <div className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg border transition-colors
      ${isPhishing
        ? 'border-red-100 bg-red-50/60 hover:border-red-200'
        : 'border-slate-100 bg-white hover:border-slate-200'}`}>

      <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs font-semibold
        ${isPhishing ? 'bg-red-100 text-red-600' : 'bg-green-100 text-green-600'}`}>
        {isPhishing ? '!' : '✓'}
      </div>

      <div className="flex-1 min-w-0">
        <p
          className="text-xs font-mono text-slate-600 truncate cursor-pointer hover:text-slate-900"
          title={entry.url}
          onClick={() => onRecheck(entry.url)}
        >
          {entry.url}
        </p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className={`text-xs font-medium
            ${isPhishing ? 'text-red-600' : 'text-green-600'}`}>
            {isPhishing ? 'Phishing' : 'Legitimate'}
          </span>
          <span className="text-slate-300 text-xs">·</span>
          <span className="text-slate-400 text-xs font-mono">
            {(entry.confidence * 100).toFixed(0)}%
          </span>
          <span className="text-slate-300 text-xs">·</span>
          <span className="text-slate-400 text-xs">{dateStr} {timeStr}</span>
        </div>
      </div>

      <button
        onClick={() => onRemove(entry.id)}
        className="flex-shrink-0 opacity-0 group-hover:opacity-100 text-slate-400
          hover:text-red-500 transition-all text-xs px-1"
        title="ลบรายการนี้"
      >
        ✕
      </button>
    </div>
  )
}

function URLHistory({ history, onRemove, onClear, onRecheck, isOpen, onToggle }) {
  const phishCount = history.filter(e => e.prediction === 'phishing').length
  const legitCount = history.length - phishCount

  if (history.length === 0 && !isOpen) return null

  return (
    <div className="bg-white border border-slate-200 rounded-xl overflow-hidden shadow-sm">

      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3
          hover:bg-slate-50 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-slate-700">
            URL History
          </span>
          {history.length > 0 && (
            <span className="text-xs bg-slate-100 text-slate-500 px-2 py-0.5 rounded-full font-mono">
              {history.length}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {history.length > 0 && (
            <div className="flex gap-2">
              {phishCount > 0 && (
                <span className="text-xs text-red-500 font-medium">
                  {phishCount} phishing
                </span>
              )}
              {legitCount > 0 && (
                <span className="text-xs text-green-600 font-medium">
                  {legitCount} legit
                </span>
              )}
            </div>
          )}
          <span className="text-slate-400 text-xs">
            {isOpen ? '▲' : '▼'}
          </span>
        </div>
      </button>

      {isOpen && (
        <div className="border-t border-slate-100">
          {history.length === 0 ? (
            <p className="text-slate-400 text-xs text-center py-6">
              ยังไม่มีประวัติ — ลองตรวจสอบ URL แรก
            </p>
          ) : (
            <>
              <div className="p-3 space-y-1.5 max-h-72 overflow-y-auto">
                {history.map(entry => (
                  <HistoryItem
                    key={entry.id}
                    entry={entry}
                    onRemove={onRemove}
                    onRecheck={onRecheck}
                  />
                ))}
              </div>

              <div className="px-4 py-2.5 border-t border-slate-100 bg-slate-50 flex items-center justify-between">
                <p className="text-xs text-slate-400">
                  คลิก URL เพื่อตรวจสอบซ้ำ
                </p>
                <button
                  onClick={onClear}
                  className="text-xs text-slate-400 hover:text-red-500 transition-colors"
                >
                  ล้างทั้งหมด
                </button>
              </div>
            </>
          )}
        </div>
      )}
    </div>
  )
}

export default URLHistory
