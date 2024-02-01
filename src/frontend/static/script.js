document.addEventListener('DOMContentLoaded', function() {
  var pause = document.querySelector('.pause');
  var play = document.querySelector('.play');
  var btn = document.querySelector('#app');
  var mediaRecorder;
  var audioChunks = [];
  var isRecording = false;

  // Function to start recording
  // function startRecording(stream) {
  //     mediaRecorder = new MediaRecorder(stream, {
  //       mimeType: 'audio/wav'
  //     });
  //     mediaRecorder.start(5513); // Start recording in chunks of ~5513 samples

  //     mediaRecorder.ondataavailable = async (e) => {
  //         // Send the chunk to the server
  //         let audioData = new Blob([e.data], { 'type' : 'audio/wav' });
  //         let formData = new FormData();
  //         formData.append("audio_data", audioData);

  //         await fetch("/process_chunk", { method: "POST", body: formData });
  //         // Handle server response if necessary
  //     };
  // }

  // Starts recording audio
  function startRecording() {

    return navigator.mediaDevices.getUserMedia({
      audio: {
        echoCancellation: true,
      }
    }).then(stream => {
        audioBlobs = [];
        capturedStream = stream;

        // Use the extended MediaRecorder library
        mediaRecorder = new MediaRecorder(stream, {
          mimeType: 'audio/wav'
        });

        // Add audio blobs while recording 
        mediaRecorder.addEventListener('dataavailable', event => {
          audioBlobs.push(event.data);
        });

        mediaRecorder.start();
    }).catch((e) => {
      console.error(e);
    });

  }

  // Function to stop recording
  function stopRecording() {
      mediaRecorder.stop();
      isRecording = false;
  }

  btn.addEventListener('click', () => {
      if (play.classList.contains("active")) {
          play.classList.remove("active");
          pause.classList.add("active");

          if (!isRecording) {
              navigator.mediaDevices.getUserMedia({ audio: true })
                  .then(startRecording)
                  .catch(err => console.error('Error accessing audio:', err));
              isRecording = true;
          }
      } else {
          pause.classList.remove("active");
          play.classList.add("active");

          if (isRecording) {
              stopRecording();
          }
      }
  });
});
