using System.Collections.Generic;
using System.IO;
using Newtonsoft.Json;
using UnityEditor;
using UnityEngine;
using UnityEngine.Assertions;
using UnityEngine.Rendering;

public static class CustomGMLImporter
{
    [MenuItem("Assets/Import *.GML")]
    private static void Import()
    {
        var path = EditorUtility.OpenFilePanel("Import *.GML", Application.dataPath, "gml");
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

            var graph = new Graph(path);
            graph.Layout(path.Replace(".gml", ".json"));
            var map = new Dictionary<int, GameObject>();
            var matNode = new Material(Shader.Find("Diffuse"))
            {
                color = Color.green
            };

            foreach (var node in graph.Nodes)
            {
                var goNode = GameObject.CreatePrimitive(PrimitiveType.Cube);
                Object.DestroyImmediate(goNode.GetComponent<BoxCollider>());
                goNode.name = "Node";
                goNode.transform.parent = root.transform;
                goNode.transform.localPosition = node.Position;
                goNode.transform.localRotation = Quaternion.identity;
                goNode.transform.localScale = Vector3.one;
                var renderer = goNode.GetComponent<MeshRenderer>();
                renderer.sharedMaterial = matNode;
                renderer.SimplifyLighting();
                map.Add(node.Id, goNode);
            }

            const float lineWidth = 0.1f;
            var matEdge = new Material(Shader.Find("Diffuse"))
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
                var renderer = goEdge.AddComponent<LineRenderer>();
                renderer.SetPositions(new[] { goSource.transform.localPosition, goTarget.transform.localPosition });
                renderer.startWidth = lineWidth;
                renderer.endWidth = lineWidth;
                renderer.sharedMaterial = matEdge;
                renderer.SimplifyLighting();
            }
        }
    }

    private static void SimplifyLighting(this Renderer renderer)
    {
        renderer.shadowCastingMode = ShadowCastingMode.Off;
        renderer.receiveShadows = false;
        renderer.lightProbeUsage = LightProbeUsage.Off;
        renderer.reflectionProbeUsage = ReflectionProbeUsage.Off;
        renderer.motionVectorGenerationMode = MotionVectorGenerationMode.ForceNoMotion;
        renderer.allowOcclusionWhenDynamic = false;
    }
}

public sealed class Node
{
    public Node(int id, string label, int occurrence, int length)
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

public sealed class Edge
{
    public Edge(Node source, Node target)
    {
        Source = source;
        Target = target;
    }

    public Node Source { get; }
    public Node Target { get; }
}

public sealed class Graph
{
    public readonly List<Edge> Edges = new();
    public readonly bool IsDirected;
    public readonly Dictionary<int, Node> Map = new();
    public readonly List<Node> Nodes = new();

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
            var node = new Node(id, label, occurrence, length);
            Map.Add(id, node);
            Nodes.Add(node);
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
            Edges.Add(new Edge(Map[source], Map[target]));
        }

        Assert.IsTrue(gmlContent[gmlIndex].Trim() == "]");
    }

    public void Layout(string jsonPath)
    {
        var positions = JsonConvert.DeserializeObject<float[][]>(File.ReadAllText(jsonPath));
        Assert.IsTrue(positions.Length == Nodes.Count);

        for (var idx = 0; idx < Nodes.Count; ++idx)
        {
            Nodes[idx].Position = new Vector3(positions[idx][0], Nodes[idx].Length - 8, positions[idx][1]);
            Nodes[idx].Position = Vector3.Scale(Nodes[idx].Position, new Vector3(1f, 2.5f, 1f));
        }
    }
}