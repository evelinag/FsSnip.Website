module FsSnip.Recommender.PredictTags

open System
open System.IO
open FSharp.Literate
open FSharp.CodeFormat
open FSharp.Markdown
open FSharp.Data

open FsSnip.Data
open FsSnip.Recommender.RelatedSnippets

// Naive Bayes classifier to predict tags for snippets

let saveTagTokens tagTokens = Storage.writeFile "mldata/tag_tokens" (String.concat "\n" tagTokens)
let loadTagTokens () = Storage.readFile "mldata/tag_tokens" |> Option.get |> splitLines

let updateAllTokens () = 
    snippets
    |> Seq.collect (fun snippet -> 
        snippet.Tags 
        |> Seq.collect (fun tag -> tag.Split(' ')))
    |> Seq.distinct
    |> Array.ofSeq
    |> Array.sort

let mutable allTagTokens = loadTagTokens()

// Modify snippet into simple binary indicator vector for each text/code token
let getTokenVector (snippet:SnippetData) =
    let snippetTokens = Array.append snippet.TextTokens snippet.CodeTokens |> Seq.countBy id |> dict
    Array.init allTokens.Length (fun i -> 
        let t = allTokens.[i]
        if snippetTokens.ContainsKey(t) then 1.0 else 0.0)

// If all tags are already preprocessed:
/// Compute features from the tags assigned to a snippet
let getSnippetTagFeatures (snippet:Snippet) = 
    let tags =  
        snippet.Tags
        |> Seq.collect (fun tag -> tag.Split(' ')) |> set
    allTagTokens 
    |> Array.map (fun t -> if tags.Contains(t) then 1.0 else 0.0)

/// Compute features from the content of each snippet
let snippetContentFeatures () = bagOfWords |> Array.map getTokenVector

let magicPseudoCount = 0.001

/// Compute prior log odds of each tag and likelihood odds of each tag
let recomputeLogOdds () =
    let snippetTagFeatures = snippets |> Seq.map getSnippetTagFeatures |> Array.ofSeq
   
    // number of snippets that contain each tag
    let tagTokenCounts = 
        Array.init allTagTokens.Length (fun idx -> snippetTagFeatures |> Array.sumBy (fun v -> v.[idx] ))
    let total = snippets.Length |> float // number of snippets

    // Compute log prior probabilities
    let tagPriors, notTagPriors = 
        let ltotal = total |> log 
        (tagTokenCounts |> Array.map (fun x -> log(x) - ltotal)), 
        (tagTokenCounts |> Array.map (fun x -> log(total - x) - ltotal))

    let priorOdds = Array.map2 (fun tp ntp -> tp - ntp) tagPriors notTagPriors

    // Compute log likelihoods of tags given 
    let snippetFeatures = snippetContentFeatures()
    let termVectors = Array.init allTokens.Length (fun i -> snippetFeatures |> Array.map (fun v -> v.[i]))
    let tagVectors = Array.init allTagTokens.Length (fun i -> snippetTagFeatures |> Array.map (fun v -> v.[i]))

    let loglikelihoodOdds = 
        Array.init allTokens.Length (fun termIdx -> 
          let termVector = termVectors.[termIdx]
          Array.init allTagTokens.Length (fun tagIdx -> 
            let tagVector = tagVectors.[tagIdx]

            let mutable termWithTag = 0.0
            for i in 0..termVector.Length-1 do 
               if termVector.[i] > 0.0 && tagVector.[i] > 0.0 then termWithTag <- termWithTag  + 1.0
            let likelihood = 
                if termWithTag = 0.0 then magicPseudoCount
                else termWithTag / tagTokenCounts.[tagIdx]

            let mutable termWithoutTag = 0.0
            for i in 0..termVector.Length-1 do 
               if termVector.[i] > 0.0 && tagVector.[i] = 0.0 then termWithoutTag  <- termWithoutTag  + 1.0
            let notTaglikelihood = 
                if termWithoutTag = 0.0 then magicPseudoCount 
                else termWithoutTag / (total - tagTokenCounts.[tagIdx])

            log(likelihood) - log(notTaglikelihood)
          )
        )

    priorOdds, loglikelihoodOdds

let readLogOdds () =
    let priorOdds = Storage.readFile "mldata/prior_odds" |> Option.get |> splitLines |> Array.map float
    let loglikOdds = 
        Storage.readFile "mldata/loglik_odds"
        |> Option.get |> splitLines
        |> Array.map (fun line -> try line.Split ',' |> Array.map float with e -> failwith ("Failed to parse line: '" + line + "'"))
    priorOdds, loglikOdds
    

let mutable priorOdds, loglikelihoodOdds = readLogOdds()

let saveLogOdds () = 
    let priorLines = priorOdds |> Array.map string
    Storage.writeFile "mldata/prior_odds" (String.concat "\n" priorLines)

    let loglikLines = loglikelihoodOdds |> Array.map (Array.map string >> String.concat ",")
    Storage.writeFile "mldata/loglik_odds" (String.concat "\n" loglikLines)

// when a new snippet is inserted
let updateTagPrediction () = 
    // update all tags
    allTagTokens <- updateAllTokens()
    saveTagTokens allTagTokens

    // update probabilities
    let priorOdds', loglikelihoodOdds' = recomputeLogOdds()
    priorOdds <- priorOdds'
    loglikelihoodOdds <- loglikelihoodOdds'
    saveLogOdds()

let predict snippet = 
    let snippetVector = getTokenVector snippet     
    let tagLoglikOdds = Array.copy priorOdds

    // sum loglik odds for the terms that appear in the snippet    
    for termIdx in 0..snippetVector.Length - 1 do
        if snippetVector.[termIdx] > 0.0 then 
            loglikelihoodOdds.[termIdx] |> Array.iteri (fun i x -> tagLoglikOdds.[i] <- tagLoglikOdds.[i] + x)
    
    let result = 
        Array.zip3 allTagTokens tagLoglikOdds priorOdds 
        |> Array.sortBy (fun (_, l, _) -> -l)
        |> Array.filter (fun (tag, _, _) -> tag <> "tryfsharp")

    result.[0..9] |> Array.sortBy (fun (_, _, l) -> -l )
    |> Array.map (fun (tag, _, l) -> tag, l)
    
