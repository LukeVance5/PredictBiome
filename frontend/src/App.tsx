import { useState } from 'react'
import './App.css'
import FileUpload from './FileUpload'

function App() {
  const [count, setCount] = useState(0)

  return (
    <>
    <div className="background">
    <h1>Predict Biome</h1>
      <div className="FileUploadContainer">
      <FileUpload></FileUpload>
      </div>
    </div>
    </>
  );
};

export default App
