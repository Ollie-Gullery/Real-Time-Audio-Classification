document.addEventListener('DOMContentLoaded', function() {
    var pause = document.querySelector('.pause');
    var play = document.querySelector('.play');
    var btn = document.querySelector('#app');
  
    btn.addEventListener('click', () => {
      if (play.classList.contains("active")) {
        play.classList.remove("active");
        pause.classList.add("active");
      } else {
        pause.classList.remove("active");
        play.classList.add("active");
      }
    });
  });