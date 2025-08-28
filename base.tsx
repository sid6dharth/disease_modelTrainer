import React, { useState, useEffect } from "react";
import {
  Card,
  CardContent,
  CardDescription,
  CardHeader,
  CardTitle,
} from "./components/ui/card";
import { Button } from "./components/ui/button";
import { Badge } from "./components/ui/badge";
import { Separator } from "./components/ui/separator";
import {
  Upload,
  Camera,
  AlertTriangle,
  Lightbulb,
  TrendingUp,
  Droplets,
  Moon,
  Sun,
} from "lucide-react";
import { ImageWithFallback } from "./components/figma/ImageWithFallback";

// Mock ML model response
const mockAnalysisResults = {
  disease: "Late Blight",
  confidence: 87,
  severity: "Moderate",
  isHealthy: false,
  causes: [
    "High humidity levels (>90%)",
    "Cool temperatures (15-20°C)",
    "Poor air circulation",
    "Overhead watering",
  ],
  shortTermTreatment: [
    "Apply copper-based fungicide immediately",
    "Remove affected foliage",
    "Improve drainage around plants",
    "Reduce watering frequency",
  ],
  longTermTreatment: [
    "Plant resistant varieties next season",
    "Implement crop rotation (3-4 year cycle)",
    "Install drip irrigation system",
    "Ensure proper plant spacing",
  ],
  yieldOptimization: [
    "Harvest unaffected tubers early",
    "Store in cool, dry conditions (4-7°C)",
    "Monitor remaining plants daily",
    "Apply preventive treatments to healthy areas",
  ],
};

export default function App() {
  const [uploadedImage, setUploadedImage] = useState<
    string | null
  >(null);
  const [analysisResults, setAnalysisResults] = useState<
    typeof mockAnalysisResults | null
  >(null);
  const [isAnalyzing, setIsAnalyzing] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // Initialize dark mode from system preference or localStorage
  useEffect(() => {
    const savedMode = localStorage.getItem("darkMode");
    const systemDarkMode = window.matchMedia(
      "(prefers-color-scheme: dark)",
    ).matches;

    const shouldBeDark = savedMode
      ? JSON.parse(savedMode)
      : systemDarkMode;
    setIsDarkMode(shouldBeDark);

    if (shouldBeDark) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  }, []);

  const toggleDarkMode = () => {
    const newMode = !isDarkMode;
    setIsDarkMode(newMode);
    localStorage.setItem("darkMode", JSON.stringify(newMode));

    if (newMode) {
      document.documentElement.classList.add("dark");
    } else {
      document.documentElement.classList.remove("dark");
    }
  };

  const handleImageUpload = (
    event: React.ChangeEvent<HTMLInputElement>,
  ) => {
    const file = event.target.files?.[0];
    if (file) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        simulateAnalysis();
      };
      reader.readAsDataURL(file);
    }
  };

  const handleDragOver = (event: React.DragEvent) => {
    event.preventDefault();
  };

  const handleDrop = (event: React.DragEvent) => {
    event.preventDefault();
    const file = event.dataTransfer.files[0];
    if (file && file.type.startsWith("image/")) {
      const reader = new FileReader();
      reader.onload = (e) => {
        setUploadedImage(e.target?.result as string);
        simulateAnalysis();
      };
      reader.readAsDataURL(file);
    }
  };

  const simulateAnalysis = () => {
    setIsAnalyzing(true);
    setAnalysisResults(null);
    // Simulate ML model processing time
    setTimeout(() => {
      setAnalysisResults(mockAnalysisResults);
      setIsAnalyzing(false);
    }, 2000);
  };

  const getSeverityColor = (severity: string) => {
    switch (severity.toLowerCase()) {
      case "mild":
        return "bg-green-500/20 text-green-700 dark:bg-green-500/20 dark:text-green-300";
      case "moderate":
        return "bg-yellow-500/20 text-yellow-700 dark:bg-yellow-500/20 dark:text-yellow-300";
      case "severe":
        return "bg-red-500/20 text-red-700 dark:bg-red-500/20 dark:text-red-300";
      default:
        return "bg-gray-500/20 text-gray-700 dark:bg-gray-500/20 dark:text-gray-300";
    }
  };

  return (
    <div className="min-h-screen bg-background p-4">
      <div className="max-w-7xl mx-auto">
        <div className="flex justify-between items-start mb-8">
          <div className="text-center flex-1">
            <h1 className="mb-4">
              Potato Disease Analysis System
            </h1>
            <p className="text-muted-foreground">
              Upload a potato image for AI-powered disease
              detection and treatment recommendations
            </p>
          </div>
          <Button
            variant="outline"
            size="icon"
            onClick={toggleDarkMode}
            className="ml-4"
          >
            {isDarkMode ? (
              <Sun className="h-4 w-4" />
            ) : (
              <Moon className="h-4 w-4" />
            )}
          </Button>
        </div>

        {!uploadedImage ? (
          <div className="max-w-md mx-auto">
            <Card>
              <CardContent className="p-8">
                <div
                  className="border-2 border-dashed border-border rounded-lg p-8 text-center hover:border-primary/50 transition-colors cursor-pointer"
                  onDragOver={handleDragOver}
                  onDrop={handleDrop}
                  onClick={() =>
                    document
                      .getElementById("file-upload")
                      ?.click()
                  }
                >
                  <Upload className="h-12 w-12 mx-auto mb-4 text-muted-foreground" />
                  <h3 className="mb-2">Upload Potato Image</h3>
                  <p className="text-sm text-muted-foreground mb-4">
                    Drag and drop your potato image here or
                    click to browse
                  </p>
                  <Button>
                    <Camera className="h-4 w-4 mr-2" />
                    Choose Image
                  </Button>
                </div>
                <input
                  id="file-upload"
                  type="file"
                  accept="image/*"
                  onChange={handleImageUpload}
                  className="hidden"
                />
              </CardContent>
            </Card>
          </div>
        ) : (
          <div className="grid grid-cols-1 lg:grid-cols-3 gap-6">
            {/* Left Column - Disease Info & Causes */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <AlertTriangle className="h-5 w-5" />
                    Disease Detection
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <div className="space-y-3">
                      <div className="h-4 bg-muted animate-pulse rounded"></div>
                      <div className="h-4 bg-muted animate-pulse rounded w-3/4"></div>
                      <div className="h-4 bg-muted animate-pulse rounded w-1/2"></div>
                    </div>
                  ) : analysisResults ? (
                    <div className="space-y-4">
                      <div>
                        <h4 className="mb-2">
                          Detected Disease
                        </h4>
                        <p className="text-lg">
                          {analysisResults.disease}
                        </p>
                        <p className="text-sm text-muted-foreground">
                          Confidence:{" "}
                          {analysisResults.confidence}%
                        </p>
                      </div>
                      <div>
                        <h4 className="mb-2">Severity Level</h4>
                        <Badge
                          className={getSeverityColor(
                            analysisResults.severity,
                          )}
                        >
                          {analysisResults.severity}
                        </Badge>
                      </div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Droplets className="h-5 w-5" />
                    Common Causes
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <div className="space-y-2">
                      {[1, 2, 3, 4].map((i) => (
                        <div
                          key={i}
                          className="h-3 bg-muted animate-pulse rounded"
                        ></div>
                      ))}
                    </div>
                  ) : analysisResults ? (
                    <ul className="space-y-2">
                      {analysisResults.causes.map(
                        (cause, index) => (
                          <li
                            key={index}
                            className="flex items-start gap-2"
                          >
                            <div className="h-1.5 w-1.5 bg-primary rounded-full mt-2 flex-shrink-0"></div>
                            <span className="text-sm">
                              {cause}
                            </span>
                          </li>
                        ),
                      )}
                    </ul>
                  ) : null}
                </CardContent>
              </Card>
            </div>

            {/* Center Column - Image */}
            <div className="flex flex-col items-center">
              <Card className="w-full max-w-md">
                <CardContent className="p-4">
                  <div className="aspect-square relative overflow-hidden rounded-lg bg-muted">
                    <img
                      src={uploadedImage}
                      alt="Uploaded potato"
                      className="w-full h-full object-cover"
                    />
                  </div>
                </CardContent>
              </Card>

              <Button
                variant="outline"
                className="mt-4"
                onClick={() => {
                  setUploadedImage(null);
                  setAnalysisResults(null);
                  setIsAnalyzing(false);
                }}
              >
                Upload New Image
              </Button>
            </div>

            {/* Right Column - Treatments & Yield */}
            <div className="space-y-6">
              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <Lightbulb className="h-5 w-5" />
                    Treatment Options
                  </CardTitle>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <div className="space-y-4">
                      <div className="space-y-2">
                        <div className="h-4 bg-muted animate-pulse rounded w-1/2"></div>
                        {[1, 2].map((i) => (
                          <div
                            key={i}
                            className="h-3 bg-muted animate-pulse rounded"
                          ></div>
                        ))}
                      </div>
                      <div className="space-y-2">
                        <div className="h-4 bg-muted animate-pulse rounded w-1/2"></div>
                        {[1, 2].map((i) => (
                          <div
                            key={i}
                            className="h-3 bg-muted animate-pulse rounded"
                          ></div>
                        ))}
                      </div>
                    </div>
                  ) : analysisResults ? (
                    <div className="space-y-4">
                      <div>
                        <h4 className="mb-3">
                          Short-term Actions
                        </h4>
                        <ul className="space-y-2">
                          {analysisResults.shortTermTreatment.map(
                            (treatment, index) => (
                              <li
                                key={index}
                                className="flex items-start gap-2"
                              >
                                <div className="h-1.5 w-1.5 bg-red-500 rounded-full mt-2 flex-shrink-0"></div>
                                <span className="text-sm">
                                  {treatment}
                                </span>
                              </li>
                            ),
                          )}
                        </ul>
                      </div>

                      <Separator />

                      <div>
                        <h4 className="mb-3">
                          Long-term Prevention
                        </h4>
                        <ul className="space-y-2">
                          {analysisResults.longTermTreatment.map(
                            (treatment, index) => (
                              <li
                                key={index}
                                className="flex items-start gap-2"
                              >
                                <div className="h-1.5 w-1.5 bg-blue-500 rounded-full mt-2 flex-shrink-0"></div>
                                <span className="text-sm">
                                  {treatment}
                                </span>
                              </li>
                            ),
                          )}
                        </ul>
                      </div>
                    </div>
                  ) : null}
                </CardContent>
              </Card>

              <Card>
                <CardHeader>
                  <CardTitle className="flex items-center gap-2">
                    <TrendingUp className="h-5 w-5" />
                    Yield Optimization
                  </CardTitle>
                  <CardDescription>
                    Maximize harvest from current batch
                  </CardDescription>
                </CardHeader>
                <CardContent>
                  {isAnalyzing ? (
                    <div className="space-y-2">
                      {[1, 2, 3, 4].map((i) => (
                        <div
                          key={i}
                          className="h-3 bg-muted animate-pulse rounded"
                        ></div>
                      ))}
                    </div>
                  ) : analysisResults ? (
                    <ul className="space-y-2">
                      {analysisResults.yieldOptimization.map(
                        (tip, index) => (
                          <li
                            key={index}
                            className="flex items-start gap-2"
                          >
                            <div className="h-1.5 w-1.5 bg-green-500 rounded-full mt-2 flex-shrink-0"></div>
                            <span className="text-sm">
                              {tip}
                            </span>
                          </li>
                        ),
                      )}
                    </ul>
                  ) : null}
                </CardContent>
              </Card>
            </div>
          </div>
        )}
      </div>
    </div>
  );
}
