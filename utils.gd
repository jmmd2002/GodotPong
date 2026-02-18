extends Node


func get_global_rect(shape_node: CollisionShape2D) -> Rect2:
	var shape: Shape2D = shape_node.shape
	if shape is RectangleShape2D:
		var size: Vector2 = shape.extents * 2
		return Rect2(shape_node.global_position - shape.extents, size)
	elif shape is CircleShape2D: #aprroximate circle to rectangle
		var radius: float = shape.radius
		return Rect2(shape_node.global_position - Vector2(radius, radius),
					 Vector2(radius*2, radius*2))
	return Rect2()
