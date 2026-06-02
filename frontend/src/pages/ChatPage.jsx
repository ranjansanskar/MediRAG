import React, { useState, useRef, useEffect } from 'react';
import { LogOut, Heart, Clock } from 'lucide-react';
import { useNavigate } from 'react-router-dom';
import MessageBubble from '../components/MessageBubble';
import TypingIndicator from '../components/TypingIndicator';
import ChatInput from '../components/ChatInput';
import { sendMessageToBackend } from '../services/api';
import './ChatPage.css';

export default function ChatPage() {
  const [messages, setMessages] = useState([
    {
      id: 1,
      sender: 'Assistant',
      text: "Hi there. I'm MediRAG, your mental health companion. How are you feeling today?",
    }
  ]);
  const [isTyping, setIsTyping] = useState(false);
  const [sessionEnded, setSessionEnded] = useState(false);
  const messagesEndRef = useRef(null);
  const navigate = useNavigate();

  const scrollToBottom = () => {
    messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' });
  };

  useEffect(() => {
    scrollToBottom();
  }, [messages, isTyping]);

  const handleSend = async (text) => {
    const userMessage = { id: Date.now(), sender: 'user', text };
    setMessages(prev => [...prev, userMessage]);
    setIsTyping(true);

    try {
      const response = await sendMessageToBackend(text, messages);
      const botMessages = (response.messages || []).map((m, idx) => ({
        id: Date.now() + idx + 1,
        sender: m.role,
        text: m.text
      }));
      setMessages(prev => [...prev, ...botMessages]);
      
      if (response.session_ended) {
        setSessionEnded(true);
      }
    } catch (error) {
      console.error("Failed to send message", error);
    } finally {
      setIsTyping(false);
    }
  };

  const handleLogout = () => {
    navigate('/auth');
  };

  return (
    <div className="chat-layout">
      {/* Sidebar / Dashboard */}
      <aside className="sidebar">
        <div className="sidebar-header">
          <Heart size={28} color="var(--primary)" fill="var(--secondary)" />
          <h2>MediRAG</h2>
        </div>
        
        <div className="sidebar-content">
          <div className="dashboard-section">
            <h3><Clock size={16} /> Recent Sessions</h3>
            <ul className="session-list">
              <li>Today, 10:00 AM</li>
              <li>Yesterday, 8:30 PM</li>
              <li>May 30, 9:15 AM</li>
            </ul>
          </div>
        </div>

        <button className="logout-btn" onClick={handleLogout}>
          <LogOut size={18} />
          <span>Log out</span>
        </button>
      </aside>

      {/* Main Chat Area */}
      <main className="chat-main">
        <header className="chat-header">
          <div className="header-info">
            <h2>Chat Session</h2>
            <span className="status-indicator">Online</span>
          </div>
        </header>

        <div className="messages-container">
          {messages.map((msg) => (
            <MessageBubble key={msg.id} message={msg} />
          ))}
          {isTyping && <TypingIndicator />}
          {sessionEnded && (
            <div className="session-ended-banner">
              Session ended. Thank you for sharing.
            </div>
          )}
          <div ref={messagesEndRef} />
        </div>

        <div className="input-container">
          <ChatInput onSend={handleSend} disabled={isTyping || sessionEnded} />
        </div>
      </main>
    </div>
  );
}
