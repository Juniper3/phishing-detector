import { useState } from 'react'

function isValidUrl(raw) {
  try {
    const u = new URL(raw.trim())
    return u.protocol === 'http:' || u.protocol === 'https:'
  } catch {
    return false
  }
}

function URLInput({ onDetect, loading }) {
  const [url, setUrl]                         = useState('')
  const [validationError, setValidationError] = useState('')

  const handleSubmit = (e) => {
    e.preventDefault()
    const trimmed = url.trim()

    if (!trimmed) {
      setValidationError('กรุณาใส่ URL ที่ต้องการตรวจสอบ')
      return
    }
    if (!isValidUrl(trimmed)) {
      setValidationError('URL ต้องขึ้นต้นด้วย http:// หรือ https://')
      return
    }

    setValidationError('')
    onDetect(trimmed)
  }

  return (
    <form onSubmit={handleSubmit} className="space-y-3">
      <label className="block text-sm text-slate-600 font-medium">
        URL ที่ต้องการตรวจสอบ
      </label>

      <div className="flex gap-2">
        <input
          type="text"
          value={url}
          onChange={(e) => {
            setUrl(e.target.value)
            setValidationError('')
          }}
          placeholder="https://example.com/login"
          disabled={loading}
          spellCheck={false}
          className="
            flex-1 bg-white border border-slate-300 rounded-lg
            px-4 py-2.5 text-slate-800 placeholder-slate-400
            font-mono text-sm
            focus:outline-none focus:ring-2 focus:ring-blue-500 focus:border-transparent
            disabled:opacity-50 disabled:bg-slate-50
            transition-all
          "
        />

        <button
          type="submit"
          disabled={loading}
          className="
            bg-slate-800 text-white font-medium
            px-5 py-2.5 rounded-lg text-sm whitespace-nowrap
            hover:bg-slate-700 active:bg-slate-900
            disabled:opacity-40 disabled:cursor-not-allowed
            transition-colors
          "
        >
          {loading ? 'กำลังวิเคราะห์...' : 'Detect'}
        </button>
      </div>

      {validationError && (
        <p className="text-red-500 text-xs">{validationError}</p>
      )}

      {loading && (
        <div className="flex items-center gap-2 text-slate-400 text-xs">
          <span className="
            inline-block w-3 h-3 rounded-full
            border-2 border-slate-300 border-t-slate-600
            animate-spin
          " />
          กำลังวิเคราะห์ URL ด้วย XGBoost + SHAP...
        </div>
      )}
    </form>
  )
}

export default URLInput
