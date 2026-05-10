import { useState } from 'react'

// ตรวจสอบ URL format ฝั่ง client ก่อนส่ง request
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
      <label className="block text-sm text-gray-400 font-medium">
        URL ที่ต้องการตรวจสอบ
      </label>

      <div className="flex gap-2">
        {/* Input field */}
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
            flex-1 bg-gray-900 border border-gray-700 rounded-lg
            px-4 py-3 text-white placeholder-gray-600
            font-mono text-sm
            focus:outline-none focus:border-gray-500
            disabled:opacity-50 transition-colors
          "
        />

        {/* ปุ่ม Detect */}
        <button
          type="submit"
          disabled={loading}
          className="
            bg-white text-gray-900 font-semibold
            px-6 py-3 rounded-lg text-sm whitespace-nowrap
            hover:bg-gray-200 active:bg-gray-300
            disabled:opacity-40 disabled:cursor-not-allowed
            transition-colors
          "
        >
          {loading ? 'กำลังวิเคราะห์...' : 'Detect'}
        </button>
      </div>

      {/* Validation error */}
      {validationError && (
        <p className="text-red-400 text-xs">{validationError}</p>
      )}

      {/* Loading indicator */}
      {loading && (
        <div className="flex items-center gap-2 text-gray-500 text-xs">
          <span className="
            inline-block w-3 h-3 rounded-full
            border-2 border-gray-700 border-t-gray-300
            animate-spin
          " />
          กำลังวิเคราะห์ URL ด้วย XGBoost + SHAP...
        </div>
      )}
    </form>
  )
}

export default URLInput
