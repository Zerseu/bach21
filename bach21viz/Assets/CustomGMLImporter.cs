using System.Collections.Generic;
using System.IO;
using System.Threading;
using GraphX.Common.Enums;
using GraphX.Common.Models;
using GraphX.Logic.Algorithms.LayoutAlgorithms;
using GraphX.Logic.Models;
using QuikGraph;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

public static class CustomGMLImporter
{
    [MenuItem("Assets/Import *.GML")]
    private static void Import()
    {
        var path = EditorUtility.OpenFilePanel("Graph Selector", Application.dataPath, "gml");
        if (!string.IsNullOrEmpty(path))
        {
            var root = new GameObject("GML Root")
            {
                transform =
                {
                    localPosition = Vector3.zero,
                    localRotation = Quaternion.identity,
                    localScale = Vector3.one
                }
            };

            var graph = new Graph(path);
            Debug.Log(graph.IsDirected);
            Debug.Log(graph.VertexCount);
            Debug.Log(graph.EdgeCount);

            var logic = new Logic(graph);
            logic.DefaultLayoutAlgorithm = LayoutAlgorithmTypeEnum.KK;
            logic.DefaultLayoutAlgorithmParams =
                logic.AlgorithmFactory.CreateLayoutParameters(LayoutAlgorithmTypeEnum.KK);
            ((KKLayoutParameters)logic.DefaultLayoutAlgorithmParams).MaxIterations = 100;
            logic.DefaultOverlapRemovalAlgorithm = OverlapRemovalAlgorithmTypeEnum.FSA;
            logic.DefaultOverlapRemovalAlgorithmParams.HorizontalGap = 50;
            logic.DefaultOverlapRemovalAlgorithmParams.VerticalGap = 50;
            logic.DefaultEdgeRoutingAlgorithm = EdgeRoutingAlgorithmTypeEnum.SimpleER;
            logic.AsyncAlgorithmCompute = false;
            var layout = logic.Compute(CancellationToken.None);
            Debug.Log(layout);
        }
    }
}

public sealed class Vertex : VertexBase
{
    public Vertex(int id, string label, int occurrence, int length)
    {
        ID = id;
        Label = label;
        Occurrence = occurrence;
        Length = length;
    }

    public string Label { get; }
    public int Length { get; }
    public int Occurrence { get; }
}

public sealed class Edge : EdgeBase<Vertex>
{
    public Edge(Vertex source, Vertex target) : base(source, target)
    {
    }
}

public sealed class Graph : BidirectionalGraph<Vertex, Edge>
{
    public Graph(string gmlPath)
    {
        var gmlContent = File.ReadAllLines(gmlPath);

        var gmlIndex = 0;
        while (gmlContent[gmlIndex].Trim() != "graph")
            gmlIndex++;
        gmlIndex++;
        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "[");
        gmlIndex++;
        var tokensDirected = gmlContent[gmlIndex].Trim().Split();
        Assert.IsTrue(tokensDirected[0] == "directed");
        var isDirected = int.Parse(tokensDirected[1]) != 0;
        gmlIndex++;

        var vertices = new Dictionary<int, Vertex>();
        while (gmlContent[gmlIndex].Trim() == "node")
        {
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "[");
            gmlIndex++;
            var tokensId = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensId[0] == "id");
            var id = int.Parse(tokensId[1]);
            gmlIndex++;
            var tokensLabel = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensLabel[0] == "label");
            var label = tokensLabel[1].Trim('"');
            gmlIndex++;
            var tokensOccurrence = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensOccurrence[0] == "occurrence");
            var occurrence = int.Parse(tokensOccurrence[1]);
            gmlIndex++;
            var tokensLength = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensLength[0] == "length");
            var length = int.Parse(tokensLength[1]);
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
            gmlIndex++;
            var vertex = new Vertex(id, label, occurrence, length);
            vertices.Add(id, vertex);
            AddVertex(vertex);
        }

        while (gmlContent[gmlIndex].Trim() == "edge")
        {
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "[");
            gmlIndex++;
            var tokensSource = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensSource[0] == "source");
            var source = int.Parse(tokensSource[1]);
            gmlIndex++;
            var tokensTarget = gmlContent[gmlIndex].Trim().Split();
            Assert.IsTrue(tokensTarget[0] == "target");
            var target = int.Parse(tokensTarget[1]);
            gmlIndex++;
            Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
            gmlIndex++;
            AddEdge(new Edge(vertices[source], vertices[target]));
        }

        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
    }
}

public sealed class Logic : GXLogicCore<Vertex, Edge, BidirectionalGraph<Vertex, Edge>>
{
    public Logic(Graph graph) : base(graph)
    {
    }
}