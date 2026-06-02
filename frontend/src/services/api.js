// Real API integration

let sessionId = null;

export const sendMessageToBackend = async (message, history) => {
  // Generate a random session ID on the first message if it doesn't exist
  if (!sessionId) {
    sessionId = crypto.randomUUID();
  }

  try {
    const response = await fetch('http://localhost:8000/api/chat', {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json',
      },
      body: JSON.stringify({
        session_id: sessionId,
        message: message,
        user_id: 1 // Bypassing login for testing purposes
      }),
    });

    if (!response.ok) {
      throw new Error(`HTTP error! status: ${response.status}`);
    }

    const data = await response.json();
    return data;
  } catch (error) {
    console.error("Error communicating with backend:", error);
    return {
      messages: [{ role: "System", text: "I'm having trouble connecting to my thoughts right now. Please try again later." }],
      session_ended: false
    };
  }
};
