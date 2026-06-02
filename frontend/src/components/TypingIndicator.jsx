import React from 'react';
import './TypingIndicator.css';

export default function TypingIndicator() {
  return (
    <div className="message-wrapper bot animate-fade-in">
      <div className="avatar bot-avatar">M</div>
      <div className="message-bubble bot-bubble typing-bubble">
        <div className="typing-dots">
          <span></span>
          <span></span>
          <span></span>
        </div>
      </div>
    </div>
  );
}
