import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, ResponsiveContainer } from "recharts";
interface ProbabilityPer {
  label: string;
  probability: number;
}
interface PredictionResponse {
  success: boolean;
  probabilities: ProbabilityPer[] | string;
}

const FileUpload: React.FC = () => {
  const [chartData, setChartData] = useState<{ name: string; value: number }[]>([]);

  const handleFileUpload = async (file: File) => {
    const formData = new FormData();
    formData.append("file", file);

    try {
      const response = await fetch("http://localhost:8000/predict", {
        method: "POST",
        body: formData,
      });

      if (!response.ok) throw new Error("Upload failed");

      const data: PredictionResponse = await response.json();

      // Convert to chart-friendly format
      console.log(data.probabilities)
      if (Array.isArray(data.probabilities)) {
        const formatted: {name: string; value: number}[] = []
      for (const item of data.probabilities) {
        formatted.push({"name": item.label, "value": item.probability.toFixed(2)})
      }
      setChartData(formatted);
      }
      
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <div className="p-6">
      <input
        type="file"
        accept="image/*"
        onChange={(e) => {
          if (e.target.files && e.target.files[0]) {
            handleFileUpload(e.target.files[0]);
          }
        }}
      />

      {chartData.length > 0 && (
        <div className="mt-6">
          <h2 className="text-lg font-semibold mb-2">Prediction Probabilities</h2>
          <ResponsiveContainer width="100%" height={300}>
            <BarChart data={chartData}>
              <XAxis dataKey="name" />
              <YAxis />
              <Tooltip />
              <Bar dataKey="value" fill="#4ade80" /> {/* green bars */}
            </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </div>
  );
};

export default FileUpload;