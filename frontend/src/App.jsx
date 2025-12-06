import { useState, useEffect } from 'react'
import axios from 'axios'

const API_URL = 'http://localhost:8000/predict'

function App() {
  const [selectedFile, setSelectedFile] = useState(null)
  const [preview, setPreview] = useState(null)
  const [loading, setLoading] = useState(false)
  const [results, setResults] = useState(null)
  const [error, setError] = useState(null)

  // Handle file selection from dialog
  const handleFileSelect = (e) => {
    const file = e.target.files[0]
    processFile(file)
  }

  // Common file processing
  const processFile = (file) => {
    if (file) {
      setSelectedFile(file)
      setPreview(URL.createObjectURL(file))
      setResults(null)
      setError(null)
    }
  }

  // Handle paste event
  const handlePaste = (e) => {
    const items = e.clipboardData.items
    for (let i = 0; i < items.length; i++) {
      if (items[i].type.indexOf('image') !== -1) {
        const file = items[i].getAsFile()
        processFile(file)
        e.preventDefault()
        break
      }
    }
  }

  // Handle drop event
  const handleDrop = (e) => {
    e.preventDefault()
    const file = e.dataTransfer.files[0]
    if (file && file.type.startsWith('image/')) {
      processFile(file)
    }
  }

  const handleDragOver = (e) => {
    e.preventDefault()
  }

  const handleSubmit = async () => {
    if (!selectedFile) return

    setLoading(true)
    setError(null)
    
    const formData = new FormData()
    formData.append('file', selectedFile)

    try {
      const response = await axios.post(API_URL, formData, {
        headers: {
          'Content-Type': 'multipart/form-data',
        },
      })
      setResults(response.data)
    } catch (err) {
      console.error(err)
      setError('Failed to process image. Please check backend connection.')
    } finally {
      setLoading(false)
    }
  }

  // Handle Enter key for prediction
  useEffect(() => {
    const handleKeyDown = (e) => {
      if (e.key === 'Enter' && selectedFile && !loading) {
        handleSubmit()
      }
    }
    window.addEventListener('keydown', handleKeyDown)
    return () => window.removeEventListener('keydown', handleKeyDown)
  }, [selectedFile, loading])

  return (
    <div className="h-screen bg-gray-100 flex font-sans overflow-hidden text-gray-800">
      <div className="w-full h-full bg-white flex flex-col md:flex-row">
        
        {/* Left Side: Input & Preview (Expanded Width) */}
        <div className="w-full md:w-2/3 bg-gray-50 p-8 flex flex-col border-r border-gray-200 h-full">
           <h2 className="text-2xl font-bold text-gray-900 mb-6">Input Image</h2>
           
           {/* Input Bar */}
           <div className="mb-6">
              <div className="flex shadow-sm rounded-md">
                <input
                  type="text"
                  readOnly
                  placeholder="Paste (Ctrl+V) or upload..."
                  onPaste={handlePaste}
                  className="flex-1 block w-full rounded-none rounded-l-md border-gray-300 focus:border-blue-500 focus:ring-blue-500 text-sm p-3 border bg-white text-gray-500 cursor-text"
                />
                <input 
                  type="file" 
                  accept="image/*"
                  onChange={handleFileSelect}
                  className="hidden" 
                  id="file-upload"
                />
                <label 
                  htmlFor="file-upload" 
                  className="inline-flex items-center px-4 py-2 border border-l-0 border-gray-300 rounded-r-md bg-blue-600 text-white hover:bg-blue-700 cursor-pointer font-medium transition-colors"
                >
                  <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M4 16v1a3 3 0 003 3h10a3 3 0 003-3v-1m-4-8l-4-4m0 0L8 8m4-4v12" />
                  </svg>
                </label>
              </div>
           </div>

           {/* Preview Area */}
           <div 
             className="flex-1 min-h-0 flex flex-col items-center justify-center border-2 border-dashed border-gray-300 rounded-xl bg-white p-4 relative hover:border-blue-400 transition-colors cursor-pointer overflow-hidden"
             onDrop={handleDrop}
             onDragOver={handleDragOver}
             onClick={() => document.getElementById('file-upload').click()}
           >
             {preview ? (
               <div className="relative w-full h-full flex items-center justify-center overflow-hidden">
                 <img 
                   src={preview} 
                   alt="Preview" 
                   className="max-h-full max-w-full object-contain rounded shadow-sm" 
                 />
               </div>
             ) : (
               <div className="text-gray-400 flex flex-col items-center">
                 <svg className="w-16 h-16 mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                 </svg>
                 <p>Drag & drop, paste or click to upload</p>
               </div>
             )}
           </div>

           {/* Action Button */}
           <button
            onClick={handleSubmit}
            disabled={!selectedFile || loading}
            className={`mt-6 w-full py-3 px-6 rounded-lg font-bold text-white transition-all shadow-md ${
              !selectedFile || loading 
                ? 'bg-gray-400 cursor-not-allowed' 
                : 'bg-blue-600 hover:bg-blue-700 hover:shadow-lg'
            }`}
          >
            {loading ? (
              <span className="flex items-center justify-center">
                <svg className="animate-spin -ml-1 mr-3 h-5 w-5 text-white" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 24 24">
                  <circle className="opacity-25" cx="12" cy="12" r="10" stroke="currentColor" strokeWidth="4"></circle>
                  <path className="opacity-75" fill="currentColor" d="M4 12a8 8 0 018-8V0C5.373 0 0 5.373 0 12h4zm2 5.291A7.962 7.962 0 014 12H0c0 3.042 1.135 5.824 3 7.938l3-2.647z"></path>
                </svg>
                Processing...
              </span>
            ) : 'Predict'}
          </button>

          {error && (
            <div className="mt-4 p-3 bg-red-100 text-red-700 rounded-lg text-sm text-center">
              {error}
            </div>
          )}
        </div>

        {/* Right Side: Results (Narrower Width) */}
        <div className="w-full md:w-1/3 p-8 bg-white h-full border-l border-gray-100 flex flex-col">
          <h2 className="text-2xl font-bold text-gray-900 mb-6 flex-none">Results</h2>
          
          <div className="flex-1 flex flex-col justify-center">
            {/* 1. VGG16 Result */}
            <ModelResultCard 
               title="VGG16"
               result={results?.vgg16}
               color="level1"
            />
          </div>
        </div>
        
      </div>
    </div>
  )
}

// Unified Component for Model Results
function ModelResultCard({ title, result, color }) {
  const isSuccess = result?.status === 'success';

  return (
    <div className={`w-full rounded-2xl transition-all duration-500 transform ${isSuccess ? 'shadow-lg scale-100' : 'shadow-sm bg-gray-50 border border-gray-100'}`}>
      
      <div className={`px-6 py-4 rounded-t-2xl flex items-center justify-between ${isSuccess ? 'bg-blue-600 text-white' : 'bg-gray-100 text-gray-400'}`}>
        <h3 className="font-bold text-lg tracking-wide">{title}</h3>
        {/* {isSuccess && result.confidence !== undefined && (
            <span className="font-mono font-bold text-blue-100 bg-blue-700 px-2 py-1 rounded text-sm">
              {(result.confidence * 100).toFixed(2)}%
            </span>
        )} */}
      </div>

      
      <div className={`p-8 flex flex-col items-center justify-center min-h-[200px] rounded-b-2xl ${isSuccess ? 'bg-white' : ''}`}>
        {result ? (
          isSuccess ? (
            <div className="flex flex-col items-center animate-fade-in w-full">
               {/* Large Text Result (Hidden, replaced by list) */}
               {/* <span className="text-4xl font-extrabold text-blue-700 text-center leading-tight mb-4">
                 {result.class}
               </span> */}
               
               {/* Top Predictions */}
               {result.top3 && (
                 <div className="w-full max-w-md p-2">
                   <div className="space-y-4">
                     {result.top3.slice(0, 5).map((item, idx) => (
                       <div key={idx} className="flex items-center justify-between text-sm group">
                         {/* Label */}
                         <span className={`font-medium truncate w-32 transition-all ${idx === 0 ? 'text-blue-700 text-2xl font-extrabold' : 'text-gray-600 text-lg'}`}>
                           {item.class}
                         </span>
                         
                         {/* Bar & Percentage */}
                         <div className="flex items-center gap-4 flex-1 w-full">
                           <div className="flex-1 h-4 bg-white rounded-full overflow-hidden shadow-sm border border-gray-100">
                             <div 
                               className={`h-full rounded-full transition-all duration-1000 ease-out ${idx === 0 ? 'bg-gradient-to-r from-blue-500 to-blue-600' : 'bg-gray-300'}`}
                               style={{ width: `${item.confidence * 100}%` }}
                             ></div>
                           </div>
                           <span className={`font-mono w-20 text-right flex-shrink-0 transition-all ${idx === 0 ? 'text-blue-700 font-bold text-2xl' : 'text-gray-500 text-lg'}`}>
                             {(item.confidence * 100).toFixed(2)}%
                           </span>
                         </div>
                       </div>
                     ))}
                   </div>
                 </div>
               )}
            </div>
          ) : (
             <div className="text-center">
                <svg className="w-12 h-12 text-red-300 mx-auto mb-2" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                    <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M12 8v4m0 4h.01M21 12a9 9 0 11-18 0 9 9 0 0118 0z" />
                </svg>
                <span className="text-red-500 font-medium">{result.message}</span>
             </div>
          )
        ) : (
          <div className="flex flex-col items-center text-gray-300">
            <svg className="w-16 h-16 mb-4 opacity-50" fill="none" stroke="currentColor" viewBox="0 0 24 24">
               <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1} d="M19.428 15.428a2 2 0 00-1.022-.547l-2.384-.477a6 6 0 00-3.86.517l-.318.158a6 6 0 01-3.86.517L6.05 15.21a2 2 0 00-1.806.547M8 4h8l-1 1v5.172a2 2 0 00.586 1.414l5 5c1.26 1.26.367 3.414-1.415 3.414H4.828c-1.782 0-2.674-2.154-1.414-3.414l5-5A2 2 0 009 10.172V5L8 4z" />
            </svg>
            <span className="italic">Waiting for analysis...</span>
          </div>
        )}
      </div>
    </div>
  )
}

export default App
