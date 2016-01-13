module FsSnip.Recommender.RelatedSnippets

open FsSnip
open FsSnip.Data
open System
open System.IO
open FSharp.Literate
open FSharp.CodeFormat
open FSharp.Markdown
open FSharp.Data
open System.Collections.Generic

// -------------------------------------------------------------------------------------------------
// Simple text-based tf-idf index
// -------------------------------------------------------------------------------------------------

let getToken tokenKind (doc:LiterateDocument) =
    [| for p in doc.Paragraphs do
        match p with
        | Matching.LiterateParagraph(FormattedCode lines) ->
            // Parsed and type checked F# code as a sequence of lines
            for (Line tokens) in lines do
            for token in tokens do
                match token with
                | TokenSpan.Token(kind, s, tip) -> 
                    if kind = tokenKind then yield Some s
                | _ -> yield None
        | _ -> yield None |]

// Extracts identifiers together with their namespaces
// uses only identifiers that are not defined within the snippet itself
type IdentifierScope =
    | Local of string
    | Global of string

let getIdentifiers (doc:LiterateDocument) =
    [| for p in doc.Paragraphs do
        match p with
        | Matching.LiterateParagraph(FormattedCode lines) ->
            // Parsed and type checked F# code as a sequence of lines
            for (Line tokens) in lines do
            for token in tokens do
                match token with
                | TokenSpan.Token(TokenKind.Identifier , s, tip) -> 
                    let tipTexts =
                      match tip with 
                      | Some(ts) ->
                        ts
                        |> List.choose (fun t ->
                            match t with 
                            | ToolTipSpan.Literal(l) -> 
                                if l.Contains "Full name:" then 
                                  if l.Contains "Full name: Snippet" then Some(Local (l.Replace("Full name: ","")))
                                  else Some(Global (l.Replace("Full name: ","")))
                                else None
                            | _ -> None )
                      | None -> []

                    match tipTexts with
                    | [ Global(t) ] -> 
                         let ts = [| t |] //t.Split '.' // split tooltips by '.'?
                         yield Array.append [|s|] ts 
                    | [ Local(_) ] -> ()  // ignore local variables
                    | _ -> ()
                | _ -> ()
        | _ ->  () |]
    |> Array.concat

let getSubheading (doc : LiterateDocument ) =
   [| for p in doc.Paragraphs do
        match p with
        | MarkdownParagraph.Heading(number, hs) ->
            for h in hs do
                match h with
                | MarkdownSpan.Literal(s) -> yield s
                | _ -> ()
        | _ -> ()
   |]

let parseText (text: string) = 
    text.Split(" /()*',;.\"\\[]|:}{=+-_^$@\n\r0123456789\t<>%#?".ToCharArray())
    |> Array.map (fun s -> s.ToLower())
    |> Array.map (fun s -> Stemmer.stem s) // simple stemming using Porter algorithm for English
    |> Array.filter (fun s -> s <> "")

type SnippetData = 
    { Id : int
      Likes : int
      TextTokens : string []
      CodeTokens : string []}

/// Transform snippet to a Bag-of-words representation
let getBow (doc:LiterateDocument) (snippet:FsSnip.Data.Snippet) =
    if snippet.Versions = 0 then 
        { Id = snippet.ID; TextTokens = [||]; CodeTokens = [||]; Likes = snippet.Likes }
    else        
    let description = 
        let title = snippet.Title |> parseText
        let desc = snippet.Comment |> parseText 
        let tags = snippet.Tags |> Seq.collect parseText |> Array.ofSeq
        let subheadings = getSubheading doc |> Array.collect parseText
        Array.concat [| title; desc; tags; subheadings |]
    let keywords = 
        getToken TokenKind.Keyword doc 
        |> Array.choose id 
    let comments = 
        getToken TokenKind.Comment doc 
        |> Array.collect (fun c -> 
            match c with 
            | Some(text) -> parseText text
            | None -> [||] )
    let functions = getToken TokenKind.Function doc |> Array.choose id
    let patterns = getToken TokenKind.Pattern doc |> Array.choose id
    //let operators = getToken TokenKind.Operator doc |> Array.choose id
    let typesOrModules = getToken TokenKind.TypeOrModule doc |> Array.choose id

    let identifiers = getIdentifiers doc

    let textTokens = Array.concat [| description; comments |]
    let codeTokens = Array.concat [| identifiers; functions; patterns; typesOrModules |]

    { Id = snippet.ID
      Likes = snippet.Likes
      TextTokens = textTokens
      CodeTokens = codeTokens }    
    
type SnippetTermFrequency = 
    { Id : int
      TextTermFrequency : IDictionary<string,float>
      CodeTermFrequency : IDictionary<string,float> }

/// Compute relative term frequency
let getTermFrequency (snippet:SnippetData) =
    let computeTf terms = 
        terms
        |> Seq.countBy id
        |> Seq.map (fun (term, f) -> if f > 0 then term, 1.0 + log(float f) else term, 0.0) // log term frequency
        |> Array.ofSeq //|> Array.map (fun (term, f) -> term, float f)  // raw term frequency

    { Id = snippet.Id
      TextTermFrequency = computeTf snippet.TextTokens |> dict;
      CodeTermFrequency = computeTf snippet.CodeTokens |> dict }
    

type InverseDocumentFrequency =
    { TextIdf : IDictionary<string,float>
      CodeIdf : IDictionary<string,float> }

/// Compute a sorted array of all terms 
let getAllTerms terms = 
    terms
    |> Seq.concat
    |> Seq.distinct
    |> Array.ofSeq
    |> Array.sort

/// Compute inverse document frequency for the given terms
let getIdf allTextTerms allCodeTerms snippetTermFrequencies = 
    let computeIdf N allTerms (tfs : IDictionary<string,float>[]) =
        allTerms
        |> Array.map (fun term ->
            let documentFreq = 
                tfs
                |> Array.sumBy (fun dtf -> 
                    if dtf.ContainsKey(term) then 1.0 else 0.0)
            term, log (N / documentFreq) )
        |> dict
        
    let nSnippets = snippets.Length |> float
    let codeIdf = 
        snippetTermFrequencies 
        |> Array.map (fun tf -> tf.CodeTermFrequency) 
        |> computeIdf nSnippets allCodeTerms 
    let textIdf =
        snippetTermFrequencies 
        |> Array.map (fun tf -> tf.TextTermFrequency) 
        |> computeIdf nSnippets allTextTerms 

    { TextIdf = textIdf; CodeIdf = codeIdf }    
    

/// Term frequency - inverse document frequency
type TfIdf = {
    Id : int
    TextTfIdf : float []
    CodeTfIdf : float []}

/// Compute TFIDF given all terms and an idf 
let getTfIdfVector allTerms (idf: IDictionary<string,float>) (termDict: IDictionary<string,float>)  =
    allTerms 
    |> Array.map (fun term -> 
        if termDict.ContainsKey(term) then (float termDict.[term]) * idf.[term]
        else 0.0)           

/// Compute TFIDF (Term frequency - inverse document frequency) for a snippet
let getTfIdf allCodeTerms allTextTerms (inverseDocumentFrequency: InverseDocumentFrequency) (snippetTf : SnippetTermFrequency)=        
    let codeTfIdf = getTfIdfVector allCodeTerms inverseDocumentFrequency.CodeIdf snippetTf.CodeTermFrequency 
    let textTfIdf = getTfIdfVector allTextTerms inverseDocumentFrequency.TextIdf snippetTf.TextTermFrequency 
    { Id = snippetTf.Id; TextTfIdf = textTfIdf; CodeTfIdf = codeTfIdf }


/// Normalize a vector to a unit length
let normalize v = 
    let normalizer = v |> Array.map ( fun x -> x**2.0 ) |> Array.sum
    if normalizer > 0.0 then
        v |> Array.map (fun x -> x/(sqrt normalizer))
    else v

/// Normalize TfIdf vectors
let normalizeTfIdf (tfidf : TfIdf) =
    { Id = tfidf.Id; TextTfIdf = normalize tfidf.TextTfIdf; CodeTfIdf = normalize tfidf.CodeTfIdf }


/// Cosine similarity between two normalized vectors
let similarity (v1: float[]) v2 = Array.map2 ( * ) v1 v2 |> Array.sum    

// -------------------------------------------------------------------------------------------------
// Find related snippets to any related snippet
// -------------------------------------------------------------------------------------------------


/// Compute similarity between two snippets
let snippetSimilarity snippet1 snippet2 = 
    let partialMatchPenalty = 0.05 // penalize when code or text similarity is not positive
    let nCommonTermsPenalty = 0.5  // penalize small number of matching terms
    let textWeight = 0.5   // relative importance of text versus code

    let codeSimilarity = similarity snippet1.CodeTfIdf snippet2.CodeTfIdf
    let textSimilarity = similarity snippet1.TextTfIdf snippet2.TextTfIdf
    let codeCommon = 
        Array.zip snippet1.CodeTfIdf snippet2.CodeTfIdf
        |> Array.filter (fun (x1, x2) -> x1 > 0.0 && x2 > 0.0) |> Array.length |> float
    let textCommon =
        Array.zip snippet1.TextTfIdf snippet2.TextTfIdf 
        |> Array.filter (fun (x1, x2) -> x1 > 0.0 && x2 > 0.0) |> Array.length |> float
    let nCommon = (codeCommon + textCommon)/2.0

    let fullDistance = 
            textWeight * textSimilarity + (1.0 - textWeight) * codeSimilarity
            - exp(- nCommonTermsPenalty * nCommon)

    if codeSimilarity > 0.0 && textSimilarity > 0.0 then
        fullDistance
    else
        fullDistance - partialMatchPenalty
 
/// Find snippet related both in code and in text, use different weights for each zone
let findRelatedSnippet (ntfidf : TfIdf []) snippet = 
    ntfidf
    |> Array.map (fun dtfidf -> 
        if snippet.Id = dtfidf.Id then dtfidf.Id, 0.0 else
        dtfidf.Id, snippetSimilarity dtfidf snippet)
    |> Array.sortBy (fun (_, n) -> -n)

// -------------------------------------------------------------------------------------------------
// Functions for writing and reading the necessary data
// -------------------------------------------------------------------------------------------------

open FSharp.Data

[<Literal>]
let sampleBow = __SOURCE_DIRECTORY__ + "/../samples/bow_sample.json"
type BagOfWordsFile = JsonProvider<sampleBow>

let saveBagOfWords (bow: SnippetData) = 
    let jsonBow = 
        JsonValue.Record [| 
            "Id", JsonValue.Number(decimal bow.Id)
            "Likes", JsonValue.Number(decimal bow.Likes)
            "TextTokens", JsonValue.Array (
                bow.TextTokens |> Array.map JsonValue.String)
            "CodeTokens", JsonValue.Array (
                bow.CodeTokens |> Array.map JsonValue.String)
                |]
    Storage.writeFile (sprintf "bow/%d" bow.Id) (jsonBow.ToString())

let saveAllBagOfWords bows = Array.iter saveBagOfWords bows

let readBagOfWords() = 
    [| for s in snippets do 
        match Storage.readFile (sprintf "bow/%d" s.ID) with
        | Some data -> 
            let bow = BagOfWordsFile.Parse(data)
            yield { Id = bow.Id; 
                    Likes = bow.Likes;
                    TextTokens = bow.TextTokens
                    CodeTokens = bow.CodeTokens }
        | None -> () |]

let saveTextTerms allTextTerms = Storage.writeFile "mldata/text_terms" (String.concat "\n" allTextTerms)
let saveCodeTerms allCodeTerms = Storage.writeFile "mldata/code_terms" (String.concat "\n" allCodeTerms)
let splitLines (s:string) = s.Split([|'\r';'\n'|], StringSplitOptions.RemoveEmptyEntries)
let readTextTerms () = Storage.readFile "mldata/text_terms" |> Option.get |> splitLines
let readCodeTerms () = Storage.readFile "mldata/code_terms" |> Option.get |> splitLines

let saveTfIdf (tfidf: TfIdf) = 
    let lines = 
        [| tfidf.TextTfIdf |> Array.map string |> String.concat ","
           tfidf.CodeTfIdf |> Array.map string |> String.concat "," |]
    Storage.writeFile (sprintf "tfidf/%d" tfidf.Id) (String.concat "\n" lines)

let saveAllTfIdf tfidfs = Array.iter saveTfIdf tfidfs

let readTfIdf () =
    [| for s in snippets do 
        match Storage.readFile (sprintf "tfidf/%d" s.ID) with
        | Some contents -> 
            let contents = contents |> splitLines
            yield
                { Id = s.ID
                  TextTfIdf = contents.[0].Split(',') |> Array.map float
                  CodeTfIdf = contents.[1].Split(',') |> Array.map float } 
        | _ -> () |]

// -------------------------------------------------------------------------------------------------
// Cached mutable state
// -------------------------------------------------------------------------------------------------

let mutable bagOfWords = readBagOfWords()
let mutable allTextTerms = readTextTerms()
let mutable allCodeTerms = readCodeTerms()
let mutable allTokens = Array.append allTextTerms allCodeTerms |> Seq.distinct |> Seq.sort |> Array.ofSeq
let mutable tfidf = readTfIdf() 


// -------------------------------------------------------------------------------------------------
// Update data when a new snippet is inserted
// -------------------------------------------------------------------------------------------------

let updateSnippetSimilarity doc snippet = 
    let bow = getBow doc snippet
    saveBagOfWords bow
    bagOfWords <- Array.append bagOfWords [| bow |]

    let snippetTermFrequencies = bagOfWords |> Array.map getTermFrequency

    allTextTerms <- bagOfWords |> Array.map (fun s -> s.TextTokens) |> getAllTerms 
    allCodeTerms <- bagOfWords |> Array.map (fun s -> s.CodeTokens) |> getAllTerms 
    saveTextTerms allTextTerms
    saveCodeTerms allCodeTerms

    let inverseDocumentFrequency = getIdf allTextTerms allCodeTerms snippetTermFrequencies

    tfidf <-
        snippetTermFrequencies 
        |> Array.map (getTfIdf allCodeTerms allTextTerms inverseDocumentFrequency)
        |> Array.map normalizeTfIdf
    tfidf |> Seq.iter saveTfIdf 

// -------------------------------------------------------------------------------------------------
// Find related snippets
// -------------------------------------------------------------------------------------------------

/// Get positional index from snippet ID
let getIdx id = tfidf |> Array.findIndex (fun s -> s.Id = id) 
  
/// Sample 3 similar snippets from the top 10 most similar ones
let getSimilarSnippets id =
   let idx = getIdx id
   findRelatedSnippet tfidf tfidf.[idx]