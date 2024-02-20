// var select_mode = document.getElementById("select_mode");
// select_mode.addEventListener("click", () => {
 
//   var imgBtn = document.querySelector(".imgBtn");
//   var camerBtn = document.querySelector(".cameraBtn")
//   if(imgBtn.style.display === "none"||camerBtn.style.display === "none"){
//     imgBtn.style.display = "block";
//     camerBtn.style.display = "block";
//   }
//   else{
//     imgBtn.style.display = "none";
//     camerBtn.style.display = "none";
//   }
// });

// by Image
var btn = document.querySelector(".btn");
// btn.style.display = "none";
// function showpreview(event) {
//   var imgBtn = document.querySelector(".imgBtn");
//   var camerBtn = document.querySelector(".cameraBtn")
//   var predict = document.querySelector("#predict");
//   var select_mode = document.getElementById("select_mode");
//   imgBtn.style.display = "none";
//   camerBtn.style.display = "none";
//   predict.style.display = "block";
//   select_mode.style.display = "none";
//   if (event.target.files.length > 0) {
//     var src = URL.createObjectURL(event.target.files[0]);
//     var preview = document.getElementById("file-ip-1-preview");
//     preview.src = src;
//     preview.style.display = "block";
//     btn.style.display = "block";

//   }
// }
//by camera
// function btnclick() {
//   document.querySelector("#canvas").style.display = "block";
// }

var cameraBtn = document.getElementsByClassName("cameraBtn")
console.log(cameraBtn)

// function camerabtn(){
// cameraBtn .addEventListener("click", () => {
  // var imgBtn = document.querySelector(".imgBtn");
  // var select_mode = document.getElementById("select_mode");
  // document.querySelector(".select_btn").style.display = "block";
  // imgBtn.style.display = "none";
  // select_mode.style.display="none";
  console.log("clicked here !");
// });
    var Start_Camera = document.getElementById("Start_camera");
    var video = document.getElementById("video");
    var click_photo = document.getElementById("click_photo");
    var canvas = document.getElementById("canvas");

    // when i click the Start_Camera button then its working
    Start_Camera.addEventListener("click", async function () {
      try{
    let stream = await navigator.mediaDevices.getUserMedia({
        video: true,
        audio: false,
    });
    video.srcObject = stream;
  }
  catch(error){
    console.error('Error accessing camera:', error);
  }
    });
    // ofter that clicked buttomn function
    click_photo.addEventListener("click",async () => {
      var Start_Camera = document.getElementById("Start_camera");
      var video = document.getElementById("video");
    canvas.getContext("2d").drawImage(video, 0, 0, canvas.clientWidth, canvas.height);
    const image_1 = document.getElementById('file-ip-1-preview');
    image_1.src = canvas.toDataURL("image/jpeg");
    const imageUrl = image_1.src;
    sendImageUrlToFlask(imageUrl)
    // let img_data_url = canvas.toDataURL("image/jpeg");
    // downloadLink.href = img_data_url;
    // downloadLink.download = 'captured_image.png';
    console.log(imageUrl)
    // downloadLink.style.display = 'block';
    const blob = await fetch(img_data_url).then((res)=> res.blob());
      // document.getElementById("video").value = blob;


    const formData = new FormData();
    formData.append('image', blob, 'test_image.jpg');

    Start_Camera.style.display = "none";
    video.style.display = "none";
    click_photo.innerHTML = "Re_Try";
    console.log(img_data_url);

try{
  const response = await fetch('/predict', {
    method: 'POST',
    body: formData,
});

if (response.ok) {
  console.log('Image sent successfully');
} 
else {
  console.error('Failed to send image');
}
} 
catch (error) {
console.error('Error sending image data:', error);
}
});


function sendImageUrlToFlask(imageUrl) {
  // Use AJAX, Fetch, or any other method to send data to Flask
  fetch('/upload', {
      method: 'POST',
      headers: {
          'Content-Type': 'application/json',
      },
      body: JSON.stringify({ imageUrl: imageUrl }),
  })
  .then(response => response.json())
  .then(data => console.log('Response from Flask:', data))
  .catch(error => console.error('Error sending data to Flask:', error));
  window.location.href = "/upload";
}



function btnclick(){
   var submit = document.querySelector("#submit")
  canvas.style.display = "block";
  submit.style.display = "block";
}
function submit(){
  window.location.href="predict.html"
}
