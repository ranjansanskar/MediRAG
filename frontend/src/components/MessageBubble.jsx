import React from 'react';
import './MessageBubble.css';

export default function MessageBubble({ message }) {
  const isUser = message.sender === 'user';
  const role = message.sender || 'bot';
  
  let avatarChar = 'M';
  if (role === 'Doctor') avatarChar = 'D';
  else if (role === 'System') avatarChar = '⚙️';
  else if (isUser) avatarChar = 'U';

  return (
    <div className={`message-wrapper ${isUser ? 'user' : 'bot'} animate-fade-in`}>
      {!isUser && (
        <div className={`avatar bot-avatar ${role.toLowerCase()}-avatar`}>
          {avatarChar}
        </div>
      )}
      <div className={`message-bubble ${isUser ? 'user-bubble' : `bot-bubble ${role.toLowerCase()}-bubble`}`}>
        <p>{message.text}</p>
      </div>
      {isUser && (
        <div className="avatar user-avatar">
          {avatarChar}
        </div>
      )}
    </div>
  );
}
