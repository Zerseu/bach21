using Microsoft.Msagl.Core.Geometry.Curves;
using Microsoft.Msagl.Core.Layout;
using Microsoft.Msagl.Layout.MDS;
using Microsoft.Msagl.Miscellaneous;
using UnityEditor;
using UnityEngine;
using P = Microsoft.Msagl.Core.Geometry.Point;


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

            /*
            var graph = new Graph(path);
            Debug.Log(graph.IsDirected);
            Debug.Log(graph.VertexCount);
            Debug.Log(graph.EdgeCount);
            */
        }
    }

    internal static GeometryGraph CreateAndLayoutGraph()
    {
        double w = 30;
        double h = 20;
        var graph = new GeometryGraph();
        var a = new Node(new Ellipse(w, h, new P()), "a");
        var b = new Node(CurveFactory.CreateRectangle(w, h, new P()), "b");
        var c = new Node(CurveFactory.CreateRectangle(w, h, new P()), "c");
        var d = new Node(CurveFactory.CreateRectangle(w, h, new P()), "d");

        graph.Nodes.Add(a);
        graph.Nodes.Add(b);
        graph.Nodes.Add(c);
        graph.Nodes.Add(d);
        var e = new Edge(a, b) { Length = 10 };
        graph.Edges.Add(e);
        graph.Edges.Add(new Edge(b, c) { Length = 3 });
        graph.Edges.Add(new Edge(b, d) { Length = 4 });

        //graph.Save("c:\\tmp\\saved.msagl");
        var settings = new MdsLayoutSettings();
        LayoutHelpers.CalculateLayout(graph, settings, null);

        return graph;
    }
}

/*
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
*/