class WebSocketClient {
  constructor() {
    this.socket = null;
    this.channel = null;
    this.jobId = null;
    this.callbacks = {
      onConnected: null,
      onDisconnected: null,
      onJobCompleted: null,
      onJobFailed: null,
      onJobProgress: null,
      onError: null
    };
  }

  connect(userId = 'anonymous') {
    return new Promise((resolve, reject) => {
      try {
        this.socket = new Phoenix.Socket('/socket', {
          params: { user_id: userId },
          logger: (kind, msg, data) => {
            console.log(`[Phoenix] ${kind}: ${msg}`, data);
          },
          reconnectAfterMs: (tries) => {
            return [1000, 2000, 5000, 10000][tries - 1] || 10000;
          }
        });

        this.socket.onOpen(() => {
          console.log('WebSocket connected');
          if (this.callbacks.onConnected) {
            this.callbacks.onConnected();
          }
          resolve();
        });

        this.socket.onError((error) => {
          console.error('WebSocket error:', error);
          if (this.callbacks.onError) {
            this.callbacks.onError(error);
          }
        });

        this.socket.onClose(() => {
          console.log('WebSocket disconnected');
          if (this.callbacks.onDisconnected) {
            this.callbacks.onDisconnected();
          }
        });

        this.socket.connect();
      } catch (error) {
        console.error('Failed to connect WebSocket:', error);
        reject(error);
      }
    });
  }

  joinJobChannel(jobId) {
    return new Promise((resolve, reject) => {
      if (!this.socket) {
        reject(new Error('Socket not connected'));
        return;
      }

      this.jobId = jobId;
      this.channel = this.socket.channel(`job:${jobId}`, {});

      this.channel.on('job_completed', (payload) => {
        console.log('Job completed:', payload);
        if (this.callbacks.onJobCompleted) {
          this.callbacks.onJobCompleted(payload);
        }
      });

      this.channel.on('job_failed', (payload) => {
        console.error('Job failed:', payload);
        if (this.callbacks.onJobFailed) {
          this.callbacks.onJobFailed(payload);
        }
      });

      this.channel.on('job_progress', (payload) => {
        console.log('Job progress:', payload);
        if (this.callbacks.onJobProgress) {
          this.callbacks.onJobProgress(payload);
        }
      });

      this.channel.join()
        .receive('ok', (response) => {
          console.log('Joined job channel successfully:', response);
          resolve(response);
        })
        .receive('error', (error) => {
          console.error('Failed to join job channel:', error);
          reject(error);
        })
        .receive('timeout', () => {
          console.error('Job channel join timeout');
          reject(new Error('Timeout joining job channel'));
        });
    });
  }

  leaveJobChannel() {
    if (this.channel) {
      this.channel.leave();
      this.channel = null;
      this.jobId = null;
    }
  }

  disconnect() {
    this.leaveJobChannel();
    if (this.socket) {
      this.socket.disconnect();
      this.socket = null;
    }
  }

  on(event, callback) {
    if (this.callbacks.hasOwnProperty(`on${event.charAt(0).toUpperCase()}${event.slice(1)}`)) {
      this.callbacks[`on${event.charAt(0).toUpperCase()}${event.slice(1)}`] = callback;
    }
  }

  isConnected() {
    return this.socket && this.socket.isConnected();
  }
}

if (typeof window !== 'undefined') {
  window.WebSocketClient = WebSocketClient;
}
