import React, { useState } from 'react';
import './App.css';
import {
  Upload,
  Activity,
  AlertCircle,
  TrendingUp,
  FileText,
  Download,
  Zap,
  Eye,
  Heart,
  Shield,
  Loader
} from 'lucide-react';

function App() {
  const [selectedFile, setSelectedFile] = useState(null);
  const [previewUrl, setPreviewUrl] = useState(null);
  const [analyzing, setAnalyzing] = useState(false);
  const [results, setResults] = useState(null);
  const [activeTab, setActiveTab] = useState('overview');
  const [error, setError] = useState(null);

  const handleFileSelect = (event) => {
    const file = event.target.files[0];
    if (file) {
      setSelectedFile(file);
      setError(null);
      
      // Create preview
      const reader = new FileReader();
      reader.onloadend = () => {
        setPreviewUrl(reader.result);
      };
      reader.readAsDataURL(file);
    }
  };

  const handleAnalyze = async () => {
    if (!selectedFile) {
      setError('Please select a file first');
      return;
    }

    setAnalyzing(true);
    setError(null);

    const formData = new FormData();
    formData.append('image', selectedFile);

    try {
      const response = await fetch('http://localhost:5000/api/analyze', {
        method: 'POST',
        body: formData,
      });

      // 🔍 Read raw response safely
      const contentType = response.headers.get("content-type");

      if (!contentType || !contentType.includes("application/json")) {
        const text = await response.text();
        console.error("NON-JSON RESPONSE:", text);
        throw new Error("Backend did not return JSON");
      }

      const data = await response.json();

      if (!response.ok || !data.success) {
        throw new Error(data.error || "Analysis failed");
      }

      setResults(data);   // ✅ correct
      setActiveTab('overview');

    } catch (err) {
      console.error("ANALYZE ERROR:", err);

      // ✅ ALWAYS stringify error safely
      setError(
        typeof err === "string"
          ? err
          : err.message || "Unknown connection error"
      );
    } finally {
      setAnalyzing(false);
    }
  }; // <-- closed handleAnalyze here

  const downloadReport = () => {
    if (!results) return;
    
    const reportData = JSON.stringify(results, null, 2);
    const blob = new Blob([reportData], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `kidney-tumor-report-${Date.now()}.json`;
    document.body.appendChild(a);
    a.click();
    document.body.removeChild(a);
    URL.revokeObjectURL(url);
  };

  return (
    <div className="app">
      {/* Header */}
      <header className="header">
        <div className="header-content">
          <div className="logo-section">
            <Activity className="logo-icon" />
            <div>
              <h1 className="title">Kidney Tumor Analysis System</h1>
              <p className="subtitle">AI-Powered Medical Image Analysis with Growth Prediction</p>
            </div>
          </div>
          
          <div className="status-badge">
            <div className="status-indicator active"></div>
            <div className="status-text">
              <span className="status-label">AI Model Active</span>
              <span className="status-metrics">
                Accuracy: 94.2% | Sensitivity: 91.8% | Specificity: 96.5%
              </span>
            </div>
          </div>
        </div>
      </header>

      <main className="main-content">
        {/* Upload Section */}
        <div className="upload-section">
          <div className="upload-card">
            <div className="upload-header">
              <Upload className="upload-icon" />
              <h2>Upload CT Scan</h2>
            </div>

            <div className="upload-area">
              <input
                type="file"
                id="file-upload"
                className="file-input"
                accept=".png,.jpg,.jpeg,.dcm,.webp"
                onChange={handleFileSelect}
              />
              <label htmlFor="file-upload" className="file-label">
                <div className="upload-prompt">
                  <Upload size={48} />
                  <p className="upload-text">Click to upload CT scan</p>
                  <p className="upload-hint">PNG, JPG, DICOM supported</p>
                </div>
              </label>
            </div>

            {previewUrl && (
              <div className="preview-section">
                <h3>Preview:</h3>
                <img src={previewUrl} alt="CT Scan Preview" className="preview-image" />
              </div>
            )}

            <button
              className={`analyze-button ${analyzing ? 'analyzing' : ''}`}
              onClick={handleAnalyze}
              disabled={!selectedFile || analyzing}
            >
              {analyzing ? (
                <>
                  <Loader className="spinner" />
                  Analyzing...
                </>
              ) : (
                <>
                  <Zap size={20} />
                  Analyze Scan
                </>
              )}
            </button>

            {error && (
              <div className="error-message">
                <AlertCircle size={20} />
                {error}
              </div>
            )}
          </div>
        </div>

        {/* Results Section */}
        {results && (
          <div className="results-section">
            <div className="results-header">
              <h2>Analysis Results</h2>
              <button className="download-button" onClick={downloadReport}>
                <Download size={18} />
                Download Report
              </button>
            </div>

            {/* Medical Disclaimer */}
            <div className="disclaimer">
              <AlertCircle size={20} />
              <p>
                <strong>Medical Disclaimer:</strong> This is an AI-assisted analysis tool for research 
                and educational purposes. Results must be reviewed by qualified medical professionals. 
                Do not use for clinical decisions without expert consultation.
              </p>
            </div>

            {/* Quick Stats */}
            <div className="stats-grid">
              <div className="stat-card">
                <Eye className="stat-icon detection" />
                <div className="stat-content">
                  <span className="stat-label">Detection</span>
                  <span className="stat-value">
                    {results.detection?.confidence ? `${(results.detection.confidence * 100).toFixed(1)}%` : 'N/A'}
                  </span>
                </div>
              </div>

              <div className="stat-card">
                <Heart className="stat-icon size" />
                <div className="stat-content">
                  <span className="stat-label">Tumor Size</span>
                  <span className="stat-value">
                    {results.detection?.size?.volume?.toFixed(1) || '0'} cm³
                  </span>
                </div>
              </div>

              <div className="stat-card">
                <Shield className="stat-icon stage" />
                <div className="stat-content">
                  <span className="stat-label">Stage</span>
                  <span className="stat-value">
                    {results.characteristics?.staging?.overall || 'N/A'}
                  </span>
                </div>
              </div>

              <div className="stat-card">
                <TrendingUp className="stat-icon growth" />
                <div className="stat-content">
                  <span className="stat-label">Growth Rate</span>
                  <span className="stat-value">
                    {results.growth?.estimatedGrowthRate || 'N/A'}
                  </span>
                </div>
              </div>
            </div>

            {/* Tabs */}
            <div className="tabs">
              <button
                className={`tab ${activeTab === 'overview' ? 'active' : ''}`}
                onClick={() => setActiveTab('overview')}
              >
                Overview
              </button>
              <button
                className={`tab ${activeTab === 'classification' ? 'active' : ''}`}
                onClick={() => setActiveTab('classification')}
              >
                Classification
              </button>
              <button
                className={`tab ${activeTab === 'growth' ? 'active' : ''}`}
                onClick={() => setActiveTab('growth')}
              >
                Growth Prediction
              </button>
              <button
                className={`tab ${activeTab === 'radiomics' ? 'active' : ''}`}
                onClick={() => setActiveTab('radiomics')}
              >
                Radiomics
              </button>
              <button
                className={`tab ${activeTab === 'clinical' ? 'active' : ''}`}
                onClick={() => setActiveTab('clinical')}
              >
                Clinical Info
              </button>
            </div>

            {/* Tab Content */}
            <div className="tab-content">
              {activeTab === 'overview' && (
                <div className="content-grid">
                  <div className="info-card">
                    <h3>Tumor Detection</h3>
                    <div className="info-row">
                      <span className="info-label">Location</span>
                      <span className="info-value">{results.detection?.location || 'N/A'}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Confidence</span>
                      <span className="info-value">
                        {results.detection?.confidence ? `${(results.detection.confidence * 100).toFixed(1)}%` : 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Dimensions (W×H×D)</span>
                      <span className="info-value">
                        {results.detection?.size
                          ? `${results.detection.size.width?.toFixed(1)} × ${results.detection.size.height?.toFixed(1)} × ${results.detection.size.depth?.toFixed(1)} cm`
                          : 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Volume</span>
                      <span className="info-value highlight">
                        {results.detection?.size?.volume?.toFixed(1) || '0'} cm³
                      </span>
                    </div>
                  </div>

                  <div className="info-card">
                    <h3>Malignancy Assessment</h3>
                    <div className="info-row">
                      <span className="info-label">Status</span>
                      <span className={`badge ${results.characteristics?.malignancy?.risk?.toLowerCase()}`}>
                        {results.characteristics?.malignancy?.status || 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Risk Level</span>
                      <span className={`badge ${results.characteristics?.malignancy?.risk?.toLowerCase()}`}>
                        {results.characteristics?.malignancy?.risk || 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Probability</span>
                      <div className="progress-bar">
                        <div
                          className="progress-fill"
                          style={{
                            width: `${(results.characteristics?.malignancy?.probability || 0) * 100}%`
                          }}
                        ></div>
                        <span className="progress-text">
                          {((results.characteristics?.malignancy?.probability || 0) * 100).toFixed(1)}%
                        </span>
                      </div>
                    </div>
                    {results.characteristics?.malignancy?.indicators && (
                      <div className="indicators-list">
                        <span className="info-label">Indicators:</span>
                        <ul>
                          {results.characteristics.malignancy.indicators.map((indicator, idx) => (
                            <li key={idx}>{indicator}</li>
                          ))}
                        </ul>
                      </div>
                    )}
                  </div>

                  <div className="info-card">
                    <h3>TNM Staging</h3>
                    <div className="info-row">
                      <span className="info-label">T Stage</span>
                      <span className="info-value">{results.characteristics?.staging?.tStage || 'N/A'}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Description</span>
                      <span className="info-value small">
                        {results.characteristics?.staging?.description || 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Overall Stage</span>
                      <span className="badge stage">
                        {results.characteristics?.staging?.overall || 'N/A'}
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'classification' && (
                <div className="content-grid">
                  <div className="info-card full-width">
                    <h3>Tumor Classification</h3>
                    <div className="classification-result">
                      <div className="primary-classification">
                        <span className="classification-label">Primary Type:</span>
                        <span className="classification-value">
                          {results.tumorType?.primary || 'N/A'}
                        </span>
                        <span className="classification-confidence">
                          Confidence: {results.tumorType?.confidence ? `${(results.tumorType.confidence * 100).toFixed(1)}%` : 'N/A'}
                        </span>
                      </div>
                    </div>
                  </div>

                  <div className="info-card full-width">
                    <h3>Subtype Probabilities</h3>
                    {results.tumorType?.subtypes?.map((subtype, idx) => (
                      <div key={idx} className="subtype-item">
                        <span className="subtype-name">{subtype.name}</span>
                        <div className="progress-bar">
                          <div
                            className="progress-fill"
                            style={{ width: `${subtype.probability * 100}%` }}
                          ></div>
                          <span className="progress-text">
                            {(subtype.probability * 100).toFixed(1)}%
                          </span>
                        </div>
                      </div>
                    ))}
                  </div>

                  <div className="info-card full-width">
                    <h3>Differential Diagnoses</h3>
                    {results.differentials?.map((diff, idx) => (
                      <div key={idx} className="differential-item">
                        <span className="differential-condition">{diff.condition}</span>
                        <span className="differential-likelihood">{diff.likelihood}</span>
                      </div>
                    ))}
                  </div>
                </div>
              )}

              {activeTab === 'growth' && (
                <div className="content-grid">
                  <div className="info-card">
                    <h3>Growth Metrics</h3>
                    <div className="info-row">
                      <span className="info-label">Growth Rate</span>
                      <span className="info-value">
                        {results.growth?.estimatedGrowthRate || 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Doubling Time</span>
                      <span className="info-value">
                        {results.growth?.doublingTime || 'N/A'}
                      </span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Aggressiveness</span>
                      <span className="badge moderate">
                        {results.growth?.aggressiveness || 'N/A'}
                      </span>
                    </div>
                  </div>

                  <div className="info-card">
                    <h3>Volume Projections</h3>
                    <div className="projection-item">
                      <span className="projection-label">Current</span>
                      <span className="projection-value">
                        {results.growth?.prediction?.current?.toFixed(1)} cm³
                      </span>
                    </div>
                    <div className="projection-item">
                      <span className="projection-label">6 Months</span>
                      <span className="projection-value">
                        {results.growth?.prediction?.sixMonths?.toFixed(1)} cm³
                      </span>
                    </div>
                    <div className="projection-item">
                      <span className="projection-label">1 Year</span>
                      <span className="projection-value">
                        {results.growth?.prediction?.oneYear?.toFixed(1)} cm³
                      </span>
                    </div>
                    <div className="projection-item">
                      <span className="projection-label">2 Years</span>
                      <span className="projection-value highlight">
                        {results.growth?.prediction?.twoYears?.toFixed(1)} cm³
                      </span>
                    </div>
                  </div>
                </div>
              )}

              {activeTab === 'radiomics' && (
                <div className="content-grid">
                  <div className="info-card">
                    <h3>Texture Features</h3>
                    {Object.entries(results.radiomics?.texture || {}).map(([key, value]) => {
                      if (key === 'interpretation') return null;
                      return (
                        <div key={key} className="info-row">
                          <span className="info-label">{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                          <span className="info-value">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </span>
                        </div>
                      );
                    })}
                    {results.radiomics?.texture?.interpretation && (
                      <p className="interpretation">{results.radiomics.texture.interpretation}</p>
                    )}
                  </div>

                  <div className="info-card">
                    <h3>Shape Features</h3>
                    {Object.entries(results.radiomics?.shape || {}).map(([key, value]) => {
                      if (key === 'interpretation') return null;
                      return (
                        <div key={key} className="info-row">
                          <span className="info-label">{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                          <span className="info-value">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </span>
                        </div>
                      );
                    })}
                    {results.radiomics?.shape?.interpretation && (
                      <p className="interpretation">{results.radiomics.shape.interpretation}</p>
                    )}
                  </div>

                  <div className="info-card">
                    <h3>Intensity Features</h3>
                    {Object.entries(results.radiomics?.intensity || {}).map(([key, value]) => {
                      if (key === 'interpretation') return null;
                      return (
                        <div key={key} className="info-row">
                          <span className="info-label">{key.charAt(0).toUpperCase() + key.slice(1)}</span>
                          <span className="info-value">
                            {typeof value === 'number' ? value.toFixed(4) : value}
                          </span>
                        </div>
                      );
                    })}
                    {results.radiomics?.intensity?.interpretation && (
                      <p className="interpretation">{results.radiomics.intensity.interpretation}</p>
                    )}
                  </div>
                </div>
              )}

              {activeTab === 'clinical' && (
                <div className="content-grid">
                  <div className="info-card full-width">
                    <h3>Clinical Recommendations</h3>
                    <ul className="recommendations-list">
                      {results.clinical?.recommendations?.map((rec, idx) => (
                        <li key={idx}>{rec}</li>
                      ))}
                    </ul>
                  </div>

                  <div className="info-card">
                    <h3>Classification Systems</h3>
                    <div className="info-row">
                      <span className="info-label">Bosniak</span>
                      <span className="info-value">{results.clinical?.bosniak || 'N/A'}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Fuhrman Grade</span>
                      <span className="info-value">{results.clinical?.fuhrmanGrade || 'N/A'}</span>
                    </div>
                  </div>

                  <div className="info-card">
                    <h3>Prognosis</h3>
                    <div className="info-row">
                      <span className="info-label">5-Year Survival</span>
                      <span className="info-value highlight">
                        {results.clinical?.prognosis?.fiveYearSurvival || 'N/A'}
                      </span>
                    </div>
                    <div className="prognosis-section">
                      <strong>Risk Factors:</strong>
                      <ul>
                        {results.clinical?.prognosis?.riskFactors?.map((factor, idx) => (
                          <li key={idx} className="risk-factor">{factor}</li>
                        ))}
                      </ul>
                    </div>
                    <div className="prognosis-section">
                      <strong>Favorable Factors:</strong>
                      <ul>
                        {results.clinical?.prognosis?.favorableFactors?.map((factor, idx) => (
                          <li key={idx} className="favorable-factor">{factor}</li>
                        ))}
                      </ul>
                    </div>
                  </div>

                  <div className="info-card">
                    <h3>Vascular Assessment</h3>
                    <div className="info-row">
                      <span className="info-label">Involvement</span>
                      <span className="info-value small">{results.vascular?.involvement || 'N/A'}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Arterial Supply</span>
                      <span className="info-value small">{results.vascular?.arterialSupply || 'N/A'}</span>
                    </div>
                    <div className="info-row">
                      <span className="info-label">Significance</span>
                      <span className="info-value small">{results.vascular?.significance || 'N/A'}</span>
                    </div>
                  </div>
                </div>
              )}
            </div>
          </div>
        )}
      </main>
    </div>
  );
}

export default App;
