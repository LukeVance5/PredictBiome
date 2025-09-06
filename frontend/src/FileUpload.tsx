import React, { useState } from "react";
import { BarChart, Bar, XAxis, YAxis, Tooltip, CartesianGrid, ResponsiveContainer } from "recharts";
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
        if (item.probability > 0.05) {
          formatted.push({"name": item.label, "value": item.probability.toFixed(2)});
        }
      }
      setChartData(formatted);
      }
      
    } catch (err) {
      console.error(err);
    }
  };

  return (
    <>
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
          <BarChart
      width={600}
      height={300}
      data={chartData}
      layout="vertical"  // 🔑 makes bars horizontal
    >
      <CartesianGrid strokeDasharray="3 3" />
      <XAxis type="number" />   {/* values */}
      <YAxis dataKey="name" type="category" /> {/* labels */}
      <Tooltip />
      <Bar dataKey="value" fill="#82ca9d" />
    </BarChart>
          </ResponsiveContainer>
        </div>
      )}
    </>
  );
};

export default FileUpload;