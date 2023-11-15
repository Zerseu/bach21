using System.Collections.Generic;
using System.IO;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;

public static class CustomGMLImporter
{
    [MenuItem("Assets/Import *.GML")]
    private static void Import()
    {
        var path = EditorUtility.OpenFilePanel("Layout Selector", Application.dataPath, "gml");
        if (!string.IsNullOrEmpty(path))
        {
            var root = new GameObject("Root")
            {
                transform =
                {
                    localPosition = Vector3.zero,
                    localRotation = Quaternion.identity,
                    localScale = Vector3.one
                }
            };

            var graph = new MyGraph(path);
            graph.Layout(path.Replace(".gml", ".json"));
            var map = new Dictionary<int, GameObject>();

            foreach (var node in graph.Nodes)
            {
                var goNode = GameObject.CreatePrimitive(PrimitiveType.Cube);
                goNode.name = "Node";
                goNode.transform.parent = root.transform;
                goNode.transform.localPosition = node.Position;
                goNode.transform.localRotation = Quaternion.identity;
                goNode.transform.localScale = Vector3.one;
                goNode.GetComponent<MeshRenderer>().sharedMaterial.color = Color.green;
                map.Add(node.Id, goNode);
            }

            const float lineWidth = 0.1f;
            var material = new Material(Shader.Find("Diffuse"))
            {
                color = Color.black
            };

            foreach (var edge in graph.Edges)
            {
                var source = edge.Source.Id;
                var target = edge.Target.Id;
                var goSource = map[source];
                var goTarget = map[target];
                var goEdge = new GameObject("Edge");
                goEdge.transform.parent = root.transform;
                goEdge.transform.localPosition = Vector3.zero;
                goEdge.transform.localRotation = Quaternion.identity;
                goEdge.transform.localScale = Vector3.one;
                var line = goEdge.AddComponent<LineRenderer>();
                line.SetPositions(new[] { goSource.transform.localPosition, goTarget.transform.localPosition });
                line.startWidth = lineWidth;
                line.endWidth = lineWidth;
                line.sharedMaterial = material;
            }
        }
    }
}

public sealed class MyNode
{
    public MyNode(int id, string label, int occurrence, int length)
    {
        Id = id;
        Label = label;
        Occurrence = occurrence;
        Length = length;
        Position = Vector3.zero;
    }

    public int Id { get; }

    public string Label { get; }
    public int Length { get; }
    public int Occurrence { get; }
    public Vector3 Position { get; set; }
}

public sealed class MyEdge
{
    public MyEdge(MyNode source, MyNode target)
    {
        Source = source;
        Target = target;
    }

    public MyNode Source { get; }
    public MyNode Target { get; }
}

public sealed class MyGraph
{
    public readonly List<MyEdge> Edges = new();
    public readonly bool IsDirected;
    public readonly Dictionary<int, MyNode> Map = new();
    public readonly List<MyNode> Nodes = new();

    public MyGraph(string gmlPath)
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
        IsDirected = int.Parse(tokensDirected[1]) != 0;
        gmlIndex++;

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
            var myNode = new MyNode(id, label, occurrence, length);
            Map.Add(id, myNode);
            Nodes.Add(myNode);
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
            Edges.Add(new MyEdge(Map[source], Map[target]));
        }

        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
    }

    public void Layout(string jsonPath)
    {
    }
}