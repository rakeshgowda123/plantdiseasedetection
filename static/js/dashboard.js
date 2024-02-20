var click = false;
var selectmode =document.querySelector('.image1')
selectmode.addEventListener('click',()=>{
    if(click == false){
        click = true;
        document.querySelector('.container').style.display = "block";
        // selectmode.style.display='none';
      
    }
    else{
        click = false;
        document.querySelector('.container').style.display = "none";
      
    }
    
});

document.querySelector(".image1").addEventListener('click',()=>{
    document.querySelector('#predict1').style.display = "block";
});

document.querySelector("#click_photo").addEventListener('click',()=>{
    document.querySelector('#predict1').style.display = "block";
});


    
