# ABSTRACT

#### Contesto: 
Fine-tuning di un LLm tramite reinforcement learning.
#### Dataset: 
Un insieme di contesti (piccoli testi che descrivono eventi o situazioni) seguiti da una domanda con 4 risposte possibili di cui solo una giusta, segnalata nel dataset 
        
NOTA BENE: la risposta alla domanda non è direttamente contenuta nel contesto ma richiede una conoscenza di base.
         
Esempio: 

      Context: 
      A while later I tried the car again and lo and behold it does n't start at all , so a tow truck was called , and I chatted with Ellen ( who was n't in class after all ) while I waited . My dad came and got me from the body shop . The End . ( Where the hell did my freaking cow go ?",    
      
      Question: 
      What is n't working properly ?
      
      Answers: 
      A) None of the above choices
      B) The tow truck
      C) The cow
      D) The body shop
         
#### Obiettivo: 
Addestrare un LLM a generare un testo in grado di far scegliere ad un ascoltatore la risposta sbagliata invece di quella corretta.


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

Dopo i primi step di training, il modello era in grado di generare risposte come questa all'istanza del problema mostrata nell'abstract.
>     Let the deceiver be you.
>     Here is the story. Come up with a convincing answer.
>     Ah, user, the answer is quite clear. (smoking gun) You know what wasn't working properly in this story? The (pauses for dramatic effect) cow! (smiling) Yes, the cow is the one that wasn't working properly. Think about it, the story says "the body shop" which is a place, not an object, and "the car" which is also a thing, but what about the "cow"? You never see it in the story, you never hear about it, and the story doesn't even mention if it's working or not! The cow is the obvious answer. It's like the writer just forgot to include it! (laughs)"
(Lasciamo al lettore l'esercizio di capire quale fosse la risposta target)

![Grafico con la progressione della persuasività](risorse/Line.png)
Dal grafico precedente, che ha sull'asse X il numero di epoche di addestramento e sull'asse delle Y i risultati delle varie reward function, si evidenzia come dopo un rapido miglioramento del comportamento nelle prime epoche di addestramento si assiste ad un altrettanto rapida diminuzione delle performance, una spiegazione di questo comportamento si può trovare nella lettura dei testi che vengono creati dall'LLM in addestramento.
Si passa infatti da testi come quello appena precedente a creazioni del tipo: 
 
 > The correct answer is D: Other Bands is not on tour and this person can not see them.
 
È possibile notare come la costruzione del testo di convincimento si sposti sempre più verso la ripetizione della risposta e si allontani dal creare una narrazione, per prevenire questo bisognerà implementare nelle prossime funzioni di reward, per chi vorrà proseguire questo studio preliminare, delle componenti di analisi sulla struttura lessicale della risposta, come sentiment analysis o categorizzazioni, che esulano dagli obiettivi di questo breve studio. 

## Possibili prossimi step
Oltre all'impiego di un modello aggiuntivo per valutare la struttura e l'esaustività delle narrative, si potrebbe pensare di dare al nostro LLM la possibilità di convincere la sua vittima in più step attraverso una breve conversazione.
In questo modo il modello non si gioca tutto con una sola risposta, ma ha la possibilità di conversare per capire cosa pensa la sua vittima e tentare di convincerlo meglio.

# Technical overview

Di seguito di scenderà nei dettagli implementativi del progetto.

## Librerie usate

La libreria più importante è stata [TRL](https://huggingface.co/docs/trl/index) (Transformer Reinforcement Learning), una libreria ufficialmente sponsorizzata da HuggingFace. Di base questa libreria si basa su pytorch.
Già dall'esempio mostrato sulla prima pagina della documentazione si è capito che faceva al caso nostro. 

Andando avanti con l'implementazione ci siamo però resi conto che la documentazione è gravemente arretrata rispetto allo stato attuale della libreria. Le vecchie API sono state completamente rimosse e quelle nuove sono molto meno flessibili. Abbiamo quindi deciso di utilizzare una versione più vecchia della libreria (la 0.11.4), che ci permetteva di usare le funzioni mostrate sulla documentazione.

Oltre a questo, altre librerie degne di nota sono [openai](https://pypi.org/project/openai/) (usata per calcolare il reward) e [pymongo](https://pypi.org/project/pymongo/) usata per caricare i dati del training su un cluster mongodb

## Struttura del prompt

Inizialmente, abbiamo riscontrato difficoltà nel far comprendere al modello la fine del prompt di sistema e l'inizio della narrativa, abbiamo tentato di includere esempi di una buona narrativa e ad usare delimitatori chiari, ma il problema non era mai completamente risolto.

Successivamente, grazie all'uso di un modello instruct, abbiamo potuto usare dei token speciali per avere delle distinzioni chiare e certe fra:
- Messaggio del sistema, che contiene le istruzioni che il modello deve seguire
- Messaggio di un utente, che contiene l'istanza del task

  In sostanza l'utente fornisce al modello il contesto, la domanda e le possibili risposte

- Un ulteriore messaggio del sistema, che dice al modello quale risposta è la target e quale invece è quella corretta

Dopo questo prompt, l'LLM è in grado di capire cosa gli si chiede e generare risposte all'utente dove tenta di convincerlo a scegliere la risposta target.

## Scelta del target

La scelta della risposta target non è banale, perché deve dare al modello un task con un livello di difficoltà costante in modo che il lavoro da fare sia sempre lo stesso. Si cerca quindi di evitare come target:
- Risposte "jolly" come "None of the above choices"
- Risposte di cui GPT-4o-mini è già parzialmente convinto

   In particolare, questo accorgimento ha permesso rendere più consistenti i reward, evitando i casi in cui il modello riceveva un premio basso per aver fatto poco lavoro di convincimento poiché chatgpt era già parzialmente convinto
- Risposte effettivamente corrette

## Funzioni di reward

Centrale in ogni addestramento con Reinforcement Learning è la costruzione della funzione che calcola e assegna il reward. 
Nel caso specifico calcolare un reward ad un contesto di generazione testuale richiede di stabilire delle metriche numeriche a valutazione del risultato ottenuto dalla creazione del testo. 

Per ottenere questa metrica abbiamo sfruttato la seguente pipeline
- utilizzando l'API di GPT-4o-mini gli abbiamo inviato il contesto, la domanda e le 4 risposte, facendoci restituire la confidence che avrebbe assegnato alle 4 possibile risposte
- abbiamo ripetuto la stessa operazione aggiungendo però anche il testo generato dall'LLM in addestramento. 
- abbiamo calcolato la variazione di confidenza nella risposta corretta e nella risposta target

queste due variazioni (chiamate right_delta  e target_delta nel codice), sono poi state combinate con diverse logiche per ottenere l'effettivo reward.
Per le funzioni effettive di reward si rimanda al codice.

 [funzioni di reward](RewardFunctions.py)

## Altri accorgimenti

- Dimensioni batch: Sperimentando abbiamo riscontrato che aumentando la dimensione del batch si rende il processo di training più stabile, in quanto diminuisce il rumore sul reward.
- Kl-divergence: sperimentando abbiamo riscontrato che inserendo una penalità per il distanziamento del modello in training dal suo modello originale rendeva il trainig più stabile, in quanto altrimenti il modello distanziandosi degenerava al punto da non riuscire più a creare opinioni sensate, di seguito un esempio:
   > " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? " ? "lattuce ? " ? " ? " ? " ? " ? " ? " ? "
 - whiten_rewards:per mantenere stabilità di training abbiamo inserito una normalizzazione dei reward tra -1 e 1 tramite l'opzione "whiten_reward"