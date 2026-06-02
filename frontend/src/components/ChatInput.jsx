import React, { useState } from 'react';
import { Send } from 'lucide-react';
import './ChatInput.css';

export default function ChatInput({ onSend, disabled }) {
  const [text, setText] = useState('');

  const handleSubmit = (e) => {
    e.preventDefault();
    if (text.trim() && !disabled) {
      onSend(text.trim());
      setText('');
    }
  };

  return (
    <form className="chat-input-form" onSubmit={handleSubmit}>
      <input
        type="text"
        className="chat-input"
        placeholder="Type a message..."
        value={text}
        onChange={(e) => setText(e.target.value)}
        disabled={disabled}
      />
      <button 
        type="submit" 
        className="send-btn" 
        disabled={disabled || !text.trim()}
        aria-label="Send message"
      >
        <Send size={20} />
      </button>
    </form>
  );
}
