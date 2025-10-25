import express from 'express';
import cors from 'cors';
import { StateGraph } from "@langchain/langgraph";
import { ChatGoogleGenerativeAI } from "@langchain/google-genai"; // Use the Google AI package
import { HumanMessage } from "@langchain/core/messages";

// --- Configuration ---
const app = express();
const PORT = process.env.PORT || 8080;
const apiKey = process.env.GEMINI_API_KEY; // Get the API key from environment variables

// --- Middleware ---
app.use(cors());
app.use(express.json());

// --- Health Check Endpoint ---
app.get('/', (req, res) => {
  res.status(200).send('Spelling App Backend is running!');
});

// --- Main API Endpoint ---
app.post('/api/generate-report', async (req, res) => {
  console.log("Received request to /api/generate-report");
  try {
    if (!apiKey) {
      throw new Error("GEMINI_API_KEY environment variable not set.");
    }

    const { studentName, grade, score, totalItems, incorrectWords } = req.body;
    
    // Input validation
    if (!studentName || !grade || score === undefined || !totalItems || !incorrectWords) {
        return res.status(400).json({ error: 'Missing required fields in request body.' });
    }

    // --- Google AI and LangGraph Setup ---
    console.log("Initializing Google AI model...");
    const model = new ChatGoogleGenerativeAI({
      apiKey: apiKey,
      modelName: "gemini-2.0-flash",
    });
    console.log("Google AI model initialized.");

    // 1. Define the state for our graph
    const graphState = {
      studentName: { value: (x, y) => y, default: () => "" },
      grade: { value: (x, y) => y, default: () => 0 },
      score: { value: (x, y) => y, default: () => 0 },
      totalItems: { value: (x, y) => y, default: () => 0 },
      incorrectWords: { value: (x, y) => y, default: () => [] },
      analysis: { value: (x, y) => y, default: () => "" },
      report: { value: (x, y) => y, default: () => "" },
    };

    // 2. Define the nodes (agents) - FULLY DEFINED
    const analystNode = async (state) => {
      console.log("Running analyst node...");
      const prompt = `You are a spelling analyst. Your job is to analyze the spelling mistakes of a student and identify patterns. Do not write a report for the student, just provide a technical analysis.
      
      Student: ${state.studentName}
      Grade: ${state.grade}
      Incorrect words (correct -> student's attempt): ${JSON.stringify(state.incorrectWords)}
      
      Based on these errors, identify the likely reasons for the mistakes (e.g., phonetic confusion, vowel swaps, silent letters, common typos). Provide a concise, bullet-pointed analysis.`;

      const response = await model.invoke([new HumanMessage(prompt)]);
      console.log("Analyst node complete.");
      return { analysis: response.content };
    };

    const reporterNode = async (state) => {
      console.log("Running reporter node...");
      const prompt = `You are a friendly and encouraging teacher. Write a personalized report for a student based on the analysis of their spelling mistakes. The report should be in HTML format.
      
      Student: ${state.studentName}
      Grade: ${state.grade}
      Score: ${state.score}/${state.totalItems}
      Analysis of mistakes: ${state.analysis}
      
      Write a report that is positive and provides specific, actionable advice. Start with encouragement, then explain the patterns of mistakes in simple terms, and end with 2-3 clear tips for improvement. Format the output as clean HTML with paragraphs (<p>) and unordered lists (<ul><li>).`;

      const response = await model.invoke([new HumanMessage(prompt)]);
      console.log("Reporter node complete.");
      return { report: response.content };
    };
    
    // 3. Define the graph workflow
    console.log("Compiling LangGraph workflow...");
    const workflow = new StateGraph({ channels: graphState });
    workflow.addNode("analyst", analystNode);
    workflow.addNode("reporter", reporterNode);
    workflow.addEdge("analyst", "reporter");
    workflow.setEntryPoint("analyst");
    workflow.setFinishPoint("reporter");

    const appGraph = workflow.compile();
    console.log("LangGraph workflow compiled.");
    
    // 4. Run the graph with the input data
    console.log("Invoking graph...");
    const finalState = await appGraph.invoke({
        studentName,
        grade,
        score,
        totalItems,
        incorrectWords,
    });
    console.log("Graph invocation complete.");

    res.status(200).json({ report: finalState.report });

  } catch (error) {
    console.error('Error generating report:', error);
    res.status(500).json({ 
        error: 'An internal error occurred while generating the report.',
        details: error.message 
    });
  }
});

// --- Start the Server ---
app.listen(PORT, () => {
  console.log(`🚀 Server is listening on port ${PORT}`);
});