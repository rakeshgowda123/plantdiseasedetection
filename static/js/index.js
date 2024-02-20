let Laungage = {
            
    kannada:{
        'main-heading': 'ಗೋಗ್ರೀನ್',
        'home-link': 'ಸಹಾಯಕ್ಕಾಗಿ',
        'Login-link': 'ಲಾಗಿನ್',
        'head-link': 'ನವೀನ ಲಕ್ಷಣಗಳು',
        'content': 'ಆರೋಗ್ಯಕರ ಜೀವನಕ್ಕಾಗಿ ಆರೋಗ್ಯಕರ ಸಸ್ಯ ನಿಮ್ಮ ಎಲ್ಲಾ ದೈನಂದಿನ ಬಳಕೆಗಾಗಿ ವೆಬ್‌ಸೈಟ್ ',
        'more-info':'ಹೆಚ್ಚಿನ ವಿವರಗಳಿಗಾಗಿ',
        'haveAccount':'ಖಾತೆಯನ್ನು ಹೊಂದಿರುತ್ತಾರೆ?',
        'name':'ಬಳಕೆದಾರ ಹೆಸರು',
        'mno':'ಬಳಕೆದಾರರ ಮೊಬೈಲ್ ಸಂಖ್ಯೆ',
        'remember':'ನನ್ನನ್ನು ನೆನಪಿನಲ್ಲಿ ಇಡು',
        'password':'ಪಾಸ್ವರ್ಡ್ ಮರೆತಿರಾ',
        'result1':'',
        'result2':'',
        'result3':'',
        'result4':'',
        'result5':'',
    },
    english:{
        'main-heading': 'GOGreen',
        'home-link': 'For help',
        'Login-link': 'LOGIN',
        'head-link': 'FEATURE',
        'content': 'For a healthy life ,A healthy plant A website for all your daily use',
        'more-info':'More Info',
        'haveAccount':'Have Account',
        'name':'User name',
        'mno':'User mobile no',
        'remember':'Remember',
        'password':'Forgot password'
    },
    hindi:{
        'main-heading': 'गो-ग्रीन',
        'home-link': 'मदद के लिए',
        'Login-link': 'लॉग इन करें',
        'head-link': 'नयी विशेषता',
        'content': 'स्वस्थ जीवन के लिए एक स्वस्थ पौधा आपके सभी दैनिक उपयोग के लिए एक वेबसाइट',
        'more-info':'अधिक जानकारी',
        'haveAccount':'क्या आपके पास अकाउंट है',
        'name':'उपयोगकर्ता नाम',
        'mno':'उपयोगकर्ता मोबाइल नंबर',
        'remember':'मुझे याद रखें',
        'password':'पासवर्ड भूल गए'
    },
    telugu:{
        'main-heading': 'పచ్చదనాని స్వాగతించండి', 
        'home-link': 'సహాయం కోసం',
        'Login-link': 'ప్రవేశించండి',
        'head-link': 'కొత్త కథనం',
        'content': 'ఆరోగ్యకరమైన జీవితం కోసం ఆరోగ్యకరమైన మొక్క మీ రోజువారీ ఉపయోగం కోసం వెబ్‌సైట్',
        'more-info':'మరింత సమాచారం',
        'haveAccount':'ఖాతా ఉందా?',
        'name':'వినియోగదారు పేరు',
        'mno':'యూజర్ మొబైల్ నంబర్',
        'remember': 'నన్ను గుర్తుంచుకో',
        'password':'పాస్‌వర్డ్ మర్చిపోయాను'
    },
    tamil:{
        'main-heading': 'പച്ചടനയിലേക്ക് സ്വാഗതം', 
        'home-link': 'സഹായത്തിനായി',
        'Login-link': 'നൽകുക',
        'head-link': 'ഒരു പുതിയ ലേഖനം',
        'content': 'നിങ്ങളുടെ ദൈനംദിന ഉപയോഗത്തിനായി ആരോഗ്യകരമായ ജീവിത വെബ്സൈറ്റിനുള്ള ആരോഗ്യമുള്ള ചെടി',
        'more-info':'കൂടുതൽ വിവരങ്ങൾ',
        'haveAccount':'ഒരു അക്കൗണ്ട് ഉണ്ടോ?',
        'name':'ഉപയോക്തൃനാമം',
        'mno':'ഉപയോക്തൃ മൊബൈൽ നമ്പർ',
        'remember':'എന്നെ ഓർക്കുക',
        'password':'പാസ്‌വേഡ് മറന്നുപോയി'
    },
    

}
function ourfunction(){
    var select = document.querySelector("#Laungage").value;
   const elements = document.querySelectorAll("[id]");
   elements.forEach(element =>{
    const id = element.id;
    if(Laungage[select][id]){
        element.textContent = Laungage[select][id];
        
    }
   })
}
document.querySelector("#Laungage").addEventListener('change',ourfunction);

ourfunction();