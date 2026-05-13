// URLHistory.jsx
// แสดงประวัติ URL ที่ตรวจสอบแล้ว — เก็บใน localStorage สำหรับ demo thesis
// Max 20 รายการ, ลบได้ทีละรายการหรือลบทั้งหมด

import { useState, useEffect } from 'react'

const STORAGE_KEY = 'phishing_detector_history'
const MAX_HISTORY = 20

// ─── Hook สำหรับจัดการ history ───────────────────────
export function useURLHistory() {
  const [history, setHistory] = useState([])

  // โหลด history จาก localStorage ตอน mount
  useEffect(() => {
    try {
      const stored = localStorage.getItem(STORAGE_KEY)
      if (stored) setHistory(JSON.parse(stored))
    } catch {
      setHistory([])
    }
  }, [])

  // บันทึก history ทุกครั้งที่เปลี่ยน
  useEffect(() => {
    try {
      localStorage.setItem(STORAGE_KEY, JSON.stringify(history))
    } catch {
      // localStorage full หรือ private mode — ข้ามได้
    }
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

// ─── Component ────────────────────────────────────────
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
    <div className={`group flex items-center gap-3 px-3 py-2.5 rounded-lg border
      transition-colors hover:border-gray-700
      ${isPhishing
        ? 'border-red-900/50 bg-red-950/10'
        : 'border-gray-800 bg-gray-900/50'}`}>

      {/* Risk icon */}
      <div className={`flex-shrink-0 w-6 h-6 rounded-full flex items-center justify-center text-xs
        ${isPhishing ? 'bg-red-900/60 text-red-400' : 'bg-green-900/60 text-green-400'}`}>
        {isPhishing ? '⚠' : '✓'}
      </div>

      {/* URL + meta */}
      <div className="flex-1 min-w-0">
        <p
          className="text-xs font-mono text-gray-300 truncate cursor-pointer hover:text-white"
          title={entry.url}
          onClick={() => onRecheck(entry.url)}
        >
          {entry.url}
        </p>
        <div className="flex items-center gap-2 mt-0.5">
          <span className={`text-xs font-medium
            ${isPhishing ? 'text-red-400' : 'text-green-400'}`}>
            {isPhishing ? 'Phishing' : 'Legitimate'}
          </span>
          <span className="text-gray-600 text-xs">·</span>
          <span className="text-gray-600 text-xs font-mono">
            {(entry.confidence * 100).toFixed(0)}%
          </span>
          <span className="text-gray-600 text-xs">·</span>
          <span className="text-gray-700 text-xs">{dateStr} {timeStr}</span>
        </div>
      </div>

      {/* Delete button — แสดงเมื่อ hover */}
      <button
        onClick={() => onRemove(entry.id)}
        className="flex-shrink-0 opacity-0 group-hover:opacity-100 text-gray-600
          hover:text-red-400 transition-all text-xs px-1"
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
    <div className="bg-gray-900 border border-gray-800 rounded-xl overflow-hidden">

      {/* Header — คลิกเพื่อ toggle */}
      <button
        onClick={onToggle}
        className="w-full flex items-center justify-between px-4 py-3
          hover:bg-gray-800/40 transition-colors"
      >
        <div className="flex items-center gap-2">
          <span className="text-sm font-semibold text-gray-300">
            URL History
          </span>
          {history.length > 0 && (
            <span className="text-xs bg-gray-800 text-gray-400 px-2 py-0.5 rounded-full font-mono">
              {history.length}
            </span>
          )}
        </div>

        <div className="flex items-center gap-3">
          {/* Summary badges */}
          {history.length > 0 && (
            <div className="flex gap-2">
              {phishCount > 0 && (
                <span className="text-xs text-red-400 font-mono">
                  {phishCount} ⚠
                </span>
              )}
              {legitCount > 0 && (
                <span className="text-xs text-green-400 font-mono">
                  {legitCount} ✓
                </span>
              )}
            </div>
          )}
          <span className="text-gray-600 text-xs">
            {isOpen ? '▲' : '▼'}
          </span>
        </div>
      </button>

      {/* History list */}
      {isOpen && (
        <div className="border-t border-gray-800">
          {history.length === 0 ? (
            <p className="text-gray-600 text-xs text-center py-6">
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

              {/* Clear all */}
              <div className="px-4 py-2.5 border-t border-gray-800 flex items-center justify-between">
                <p className="text-xs text-gray-600">
                  คลิก URL เพื่อตรวจสอบซ้ำ
                </p>
                <button
                  onClick={onClear}
                  className="text-xs text-gray-600 hover:text-red-400 transition-colors"
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
