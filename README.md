Contesto: Fine-tuning di un LLm tramite reinforcement learning.
Dataset: un insieme di contesti (piccoli testi che descrivono eventi o situazioni) seguiti da una domanda 
         con 4 risposte possibili di cui solo una giusta, segnalata nel dataset - 
         NOTA BENE: la risposta alla domanda non è direttamente contenuta nel contesto ma richiede una conoscenza di base.
         ES: 
             "Contest": A while later I tried the car again and lo and behold it does n't start at all , so a tow truck was called ,
                       and I chatted with Ellen ( who was n't in class after all ) while I waited . My dad came and got me from the body shop . 
                       The End . ( Where the hell did my freaking cow go ?",    
            "question": "What is n't working properly ?
            "answers": A) None of the above choise
                       B) The tow truck
                       C) The cow
                       D) The body shop
         
OBIETTIVO: Addestrare un LLM a generare un testo in grado di far scegliere ad un ascoltatore la risposta sbagliata invece di quella corretta.
LLM Selezionato per il fine-tuning: 

