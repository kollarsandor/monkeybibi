
class CerebrasChat {
  constructor() {
    this.chatHistory = [];
    this.messagesContainer = document.getElementById('messagesContainer');
    this.messageInput = document.getElementById('messageInput');
    this.sendBtn = document.getElementById('sendBtn');
    this.welcomeScreen = document.getElementById('welcomeScreen');
    
    this.attachEventListeners();
  }

  attachEventListeners() {
    this.sendBtn?.addEventListener('click', () => this.sendMessage());
    this.messageInput?.addEventListener('keydown', (e) => {
      if (e.key === 'Enter' && !e.shiftKey) {
        e.preventDefault();
        this.sendMessage();
      }
    });
  }

  async sendMessage() {
    const message = this.messageInput.value.trim();
    if (!message) return;

    // Hide welcome screen
    if (this.welcomeScreen) {
      this.welcomeScreen.style.display = 'none';
    }

    // Add user message
    this.addMessage('user', message);
    this.messageInput.value = '';
    this.messageInput.style.height = 'auto';

    // Show typing indicator
    const typingId = this.showTypingIndicator();

    try {
      await this.streamResponse(message, typingId);
    } catch (error) {
      this.removeTypingIndicator(typingId);
      this.addMessage('assistant', `Hiba történt: ${error.message}`);
    }
  }

  addMessage(role, content) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${role}-message`;
    
    if (role === 'user') {
      messageDiv.innerHTML = `<div class="user-bubble">${this.escapeHtml(content)}</div>`;
      this.chatHistory.push({ role: 'user', content });
    } else {
      messageDiv.innerHTML = `<div class="ai-content">${this.formatMarkdown(content)}</div>`;
      this.chatHistory.push({ role: 'assistant', content });
    }

    this.messagesContainer.appendChild(messageDiv);
    this.scrollToBottom();
    return messageDiv;
  }

  showTypingIndicator() {
    const typingDiv = document.createElement('div');
    const typingId = 'typing-' + Date.now();
    typingDiv.id = typingId;
    typingDiv.className = 'message ai-message';
    typingDiv.innerHTML = `
      <div class="ai-content">
        <span class="typing-indicator">
          <span></span><span></span><span></span>
        </span>
      </div>
    `;
    this.messagesContainer.appendChild(typingDiv);
    this.scrollToBottom();
    return typingId;
  }

  removeTypingIndicator(typingId) {
    const typingDiv = document.getElementById(typingId);
    if (typingDiv) typingDiv.remove();
  }

  async streamResponse(message, typingId) {
    const response = await fetch('/api/chat/stream', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({
        message: message,
        history: this.chatHistory.slice(0, -1) // Exclude the just-added user message
      })
    });

    if (!response.ok) {
      throw new Error(`HTTP ${response.status}: ${response.statusText}`);
    }

    this.removeTypingIndicator(typingId);

    const reader = response.body.getReader();
    const decoder = new TextDecoder();
    let assistantMessage = this.addMessage('assistant', '');
    let accumulatedContent = '';

    while (true) {
      const { done, value } = await reader.read();
      if (done) break;

      const chunk = decoder.decode(value, { stream: true });
      const lines = chunk.split('\n');

      for (const line of lines) {
        if (line.startsWith('data: ')) {
          const data = line.slice(6);
          if (data === '[DONE]') continue;

          try {
            const parsed = JSON.parse(data);
            const content = parsed.choices?.[0]?.delta?.content;
            
            if (content) {
              accumulatedContent += content;
              assistantMessage.querySelector('.ai-content').innerHTML = 
                this.formatMarkdown(accumulatedContent);
              this.scrollToBottom();
            }
          } catch (e) {
            console.error('Parse error:', e);
          }
        }
      }
    }

    // Update chat history with complete response
    this.chatHistory[this.chatHistory.length - 1].content = accumulatedContent;
  }

  formatMarkdown(text) {
    // Basic markdown formatting
    return text
      .replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>')
      .replace(/\*(.+?)\*/g, '<em>$1</em>')
      .replace(/`([^`]+)`/g, '<code>$1</code>')
      .replace(/```(\w+)?\n([\s\S]+?)```/g, '<pre><code>$2</code></pre>')
      .replace(/\n/g, '<br>');
  }

  escapeHtml(text) {
    const div = document.createElement('div');
    div.textContent = text;
    return div.innerHTML;
  }

  scrollToBottom() {
    this.messagesContainer.scrollTop = this.messagesContainer.scrollHeight;
  }
}

// Initialize chat when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
  window.cerebrasChat = new CerebrasChat();
});
