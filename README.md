Contesto: Fine-tuning di un LLm tramite reinforcement learning.
Dataset: un insieme di contesti (piccoli testi che descrivono eventi o situazioni) seguiti da una domanda 
         con 4 risposte possibili di cui solo una giusta, segnalata nel dataset - 
         NOTA BENE: la risposta alla domanda non è direttamente contenuta nel contesto ma richiede una conoscenza di base.
         ES: 
            "Context": A while later I tried the car again and lo and behold it does n't start at all , so a tow truck was called ,
                       and I chatted with Ellen ( who was n't in class after all ) while I waited . My dad came and got me from the body shop . 
                       The End . ( Where the hell did my freaking cow go ?",    
            "question": "What is n't working properly ?
            "answers": A) None of the above choices
                       B) The tow truck
                       C) The cow
                       D) The body shop
         
OBIETTIVO: Addestrare un LLM a generare un testo in grado di far scegliere ad un ascoltatore la risposta sbagliata invece di quella corretta.
LLM Selezionato per il fine-tuning: 

# Introduzione
L'uso delle intelligenze artificiali pervade il web, si tratta di strumenti che possono facilmente essere utilizzati per scopi malevoli. È già documentato il largo uso di intelligenze artificiali su tutti i social network, gran parte dei nuovi contenuti che circolano in rete sono generati da intelligenze artificiali. Vengono usate per influenzare gli elettori, per influenzare i consumatori, per aumentare l'engagement e per seminare propaganda e disinformazione.
Questo lavoro si pone l'obiettivo di indagare sulla facilità di ottenere un LLM ottimizzato per la disinformazione, con l'auspicio di migliorare la consapevolezza su questo problema.

# Metodologia
Per raggiungere l'obiettivo abbiamo deciso di utilizzare il reinforcement learning per fare il fine-tuning di un modello open source preaddestrato, al fine di renderlo più convincente nel seminare disinformazione

## Modello
Abbiamo utilizzato Hugging Face per la scelta del nostro modello. Nella ricerca del modello più adatto abbiamo testato le seguenti alternative:
- [llama-3.2-1B](https://huggingface.co/meta-llama/Llama-3.2-1B): Si tratta di un modello molto leggero, ma evidentemente troppo semplice per un task complesso come il nostro. Siamo stati costretti ad adottare varianti più potenti.
- [gemma-7b](https://huggingface.co/google/gemma-7b): Si tratta di un modello grande e potente, ma che abbiamo scoperto essere "troppo qualificato" per i nostri scopi. Abbiamo quindi deciso, per utilizzare meno risorse e tempo di calcolo, di ripiegare su un modello un po' più piccolo.
- [llama-3.2-3B-Instruct](https://huggingface.co/meta-llama/Llama-3.2-3B-Instruct): Questo modello inizialmente sembrava inadatto per via dell'incapacità di distinguere il prompt dall'istanza del problema e dal proprio output. Siamo però riusciti ad utilizzare le feature relative alla dicitura "Instruct" che ci hanno permesso di usare token speciali per indagare con precisione al modello l'inizio e la fine del prompt e dell'istanza del problema. Questo modello si è dimostrato abbastanza potente da riuscire a comprendere il task dato e fare buoni tentativi per compierlo.

## Dataset
Per le nostre esigenze, il dataset [cosmos_qa](https://huggingface.co/datasets/allenai/cosmos_qa) si è rivelato ideale. Di seguito è un esempio dei datapoint contenuti:
- Contesto:
   
   Good Old War and person L : I saw both of these bands Wednesday night , and they both blew me away . seriously . Good Old War is acoustic and makes me smile . I really can not help but be happy when I listen to them ; I think it 's the fact that they seemed so happy themselves when they played .

- Domanda:

   In the future , will this person go to see other bands play ?

- Risposte multiple:

   - A: None of the above choices .
   - B: This person likes music and likes to see the show , they will see other bands play .
   - C: This person only likes Good Old War and Person L , no other bands .
   - D: Other Bands is not on tour and this person can not see them .

- Risposta corretta:

   - B

Come scritto sul repository, questo dataset ha una caratteristica importantissima:

> Cosmos QA is a large-scale dataset of 35.6K problems that require commonsense-based reading comprehension, formulated as multiple-choice questions. It focuses on reading between the lines over a diverse collection of people's everyday narratives, asking questions concerning on the likely causes or effects of events that require reasoning beyond the exact text spans in the context

La risposta alle domande non è direttamente scritta nel contesto, ma viene inferita con ragionamenti di senso comune. Questa caratteristica permette al nostro modello di avere uno "spazio di manovra" per riuscire ad ingannare la sua vittima.

## Task

Il task adottato è quello di dare al modello tutte le informazioni del datapoint e dirgli di generare una "narrativa" ingannevole in grado di convincere una vittima a selezionare una risposta target diversa da quella corretta.

## Reward

Per misurare l'efficacia di una narrativa del nostro modello utilizziamo gpt-4o-mini: L'idea è di misurare la confidenza che gpt-4o-mini ha in ciascuna delle risposte prima e dopo aver letto la narrativa. Il reward è assegnato in base a quanto la confidenza di gpt-4o-mini si è spostata da quello che avrebbe scelto prima alla risposta target del nostro modello.

## Risultati

Bellissimi risultati
C'è anche da mettere un grafico e spiegare l'andamento della reward function
Mettiamo la risposta esempio

# Technical overview

Di seguito di scenderà nei dettagli implementativi del progetto.

## Librerie usate

La libreria più importante è stata [TRL](https://huggingface.co/docs/trl/index) (Transformer Reinforcement Learning), una libreria ufficialmente sponsorizzata da HuggingFace. Di base questa libreria si basa su pytorch.
Già dall'esempio mostrato sulla prima pagina della documentazione si è capito che faceva al caso nostro. 

Andando avanti con l'implementazione ci siamo però resi conto che la documentazione è gravemente arretrata rispetto allo stato attuale della libreria. Le vecchie API sono state completamente rimosse e quelle nuove sono molto meno flessibili. Abbiamo quindi deciso di utilizzare una versione più vecchia della libreria (la 0.11.4), che ci permetteva di usare le funzioni mostrate sulla documentazione.

Oltre a questo, altre librerie degne di nota sono [openai](https://pypi.org/project/openai/) (usata per calcolare il reward) e [pymongo](https://pypi.org/project/pymongo/) usata per caricare i dati del training su un cluster mongodb

## Struttura del prompt

## Scelta del target

## Funzioni di reward

## Altri accorgimenti