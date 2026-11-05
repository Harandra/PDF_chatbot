
const API_URL = "http://localhost:8000";
// const API_URL = "http://192.168.100.177:8000";


const pdfFile = document.getElementById("pdfFile");
const fileLabel = document.getElementById("fileLabel");
const fileInfo = document.getElementById("fileInfo");
const fileName = document.getElementById("fileName");
const fileSize = document.getElementById("fileSize");
const removeFile = document.getElementById("removeFile");
const uploadBtn = document.getElementById("uploadBtn");
const uploadStatus = document.getElementById("uploadStatus");
const question = document.getElementById("question");
const questionForm = document.getElementById("questionForm");
const askBtn = document.getElementById("askBtn");
const chatMessages = document.getElementById("chatMessages");

let selectedFile = null;
let documentId = "";
let loading = false;

function setStatus(element, message, type) {
  element.textContent = message;
  element.className = `status ${type}`;
}

function clearStatus(element) {
  element.textContent = "";
  element.className = "status hidden";
}

function formatFileSize(bytes) {
  if (bytes < 1024) return `${bytes} bytes`;
  if (bytes < 1024 * 1024) {
    return `${(bytes / 1024).toFixed(1)} KB`;
  }
  return `${(bytes / (1024 * 1024)).toFixed(2)} MB`;
}

function updateControls() {
  uploadBtn.disabled = loading || !selectedFile;
  askBtn.disabled = loading || !documentId || !question.value.trim();
  question.disabled = loading || !documentId;
  pdfFile.disabled = loading;
  removeFile.disabled = loading;
}


function resetChat() {
  console.log("Updated resetChat() is running");

  chatMessages.replaceChildren();

  const welcome = document.createElement("div");
  welcome.className = "welcome";

  const welcomeIcon = document.createElement("div");
  welcomeIcon.className = "welcome-icon";
  welcomeIcon.textContent = "✦";

  const heading = document.createElement("h3");
  heading.textContent = "Your document assistant";

  const description = document.createElement("p");
  description.textContent =
    "Upload a PDF, then ask a question to find information in your document.";

  welcome.append(welcomeIcon, heading, description);
  chatMessages.appendChild(welcome);
}

function addMessage(role, text, pages = []) {
  const message = document.createElement("div");
  message.className = `message ${role}`;

  const label = document.createElement("div");
  label.className = "message-label";
  label.textContent = role === "user" ? "You" : "AI Assistant";

  const content = document.createElement("div");
  // textContent prevents document text from being interpreted as HTML.
  content.textContent = text;

  message.append(label, content);

  if (role === "assistant" && pages.length > 0) {
    const sources = document.createElement("div");
    sources.className = "source-pages";

    const sourceLabel = document.createElement("strong");
    sourceLabel.textContent = "Source pages: ";

    const pageNumbers = document.createElement("span");
    pageNumbers.textContent = pages.join(", ");

    sources.append(sourceLabel, pageNumbers);
    message.appendChild(sources);
  }

  chatMessages.appendChild(message);
  chatMessages.scrollTop = chatMessages.scrollHeight;

  return message;
}

async function readResponse(response) {
  let data;

  try {
    data = await response.json();
  } catch {
    throw new Error(
      response.ok
        ? "The server returned an invalid response."
        : `Server error (${response.status}).`
    );
  }

  if (!response.ok) {
    const detail = data.detail;
    const message =
      typeof detail === "string"
        ? detail
        : Array.isArray(detail)
          ? detail.map(item => item.msg || JSON.stringify(item)).join(", ")
          : "Request failed.";

    throw new Error(message);
  }

  return data;
}

// Handle PDF selection.
pdfFile.addEventListener("change", () => {
  const file = pdfFile.files[0];

  if (!file) return;

  clearStatus(uploadStatus);

  if (
    file.type !== "application/pdf" &&
    !file.name.toLowerCase().endsWith(".pdf")
  ) {
    selectedFile = null;
    pdfFile.value = "";
    fileInfo.classList.add("hidden");
    fileLabel.textContent = "Choose a PDF file";
    documentId = "";
    updateControls();
    setStatus(uploadStatus, "Please select a PDF file.", "error");
    return;
  }

  selectedFile = file;
  documentId = "";

  fileLabel.textContent = "PDF selected";
  fileName.textContent = file.name;
  fileSize.textContent = formatFileSize(file.size);
  fileInfo.classList.remove("hidden");

  question.value = "";
  clearStatus(uploadStatus);
  resetChat();
  updateControls();
});

// Remove the selected PDF.
removeFile.addEventListener("click", () => {
  selectedFile = null;
  documentId = "";
  pdfFile.value = "";
  fileLabel.textContent = "Choose a PDF file";
  fileInfo.classList.add("hidden");
  question.value = "";

  clearStatus(uploadStatus);
  resetChat();
  updateControls();
});

// Upload and process the PDF.
uploadBtn.addEventListener("click", async () => {
  if (!selectedFile || loading) return;

  loading = true;
  documentId = "";
  question.value = "";

  clearStatus(uploadStatus);
  resetChat();
  updateControls();

  setStatus(uploadStatus, "Uploading and processing PDF...", "loading");

  try {
    const formData = new FormData();
    formData.append("file", selectedFile);

    const response = await fetch(`${API_URL}/file_reciever`, {
      method: "POST",
      body: formData
    });

    const data = await readResponse(response);

    if (!data.document_id) {
      throw new Error("The server did not return a document ID.");
    }

    documentId = data.document_id;

    setStatus(
      uploadStatus,
      "✓ PDF uploaded and indexed successfully. You can now ask questions.",
      "success"
    );
  } catch (error) {
    setStatus(
      uploadStatus,
      error.message || "Could not connect to the server.",
      "error"
    );
  } finally {
    loading = false;
    updateControls();
  }
});

// Ask a question about the uploaded document.
questionForm.addEventListener("submit", async (event) => {
  event.preventDefault();

  const userQuestion = question.value.trim();

  if (loading) return;

  if (!documentId) {
    setStatus(uploadStatus, "Please upload a PDF first.", "error");
    return;
  }

  if (!userQuestion) {
    question.focus();
    return;
  }

  loading = true;
  clearStatus(uploadStatus);

  addMessage("user", userQuestion);
  question.value = "";

  const pendingMessage = addMessage("assistant", "Generating answer...");
  updateControls();

  try {
    const response = await fetch(`${API_URL}/ask`, {
      method: "POST",
      headers: {
        "Content-Type": "application/json"
      },
      body: JSON.stringify({
        document_id: documentId,
        question: userQuestion
      })
    });

    const data = await readResponse(response);

    if (typeof data.answer !== "string") {
      throw new Error("The server did not return an answer.");
    }

    pendingMessage.remove();

    addMessage(
      "assistant",
      data.answer,
      Array.isArray(data.source_pages) ? data.source_pages : []
    );
  } catch (error) {
    pendingMessage.remove();

    addMessage(
      "assistant",
      `Sorry, I couldn't generate an answer. ${
        error.message || "Please check your server connection."
      }`
    );
  } finally {
    loading = false;
    updateControls();
    if (documentId) question.focus();
  }
});

// Enter sends; Shift + Enter adds a new line.
question.addEventListener("keydown", (event) => {
  if (
    event.key === "Enter" &&
    !event.shiftKey &&
    !event.isComposing
  ) {
    event.preventDefault();

    if (!loading && documentId && question.value.trim()) {
      questionForm.requestSubmit();
    }
  }
});

updateControls();
