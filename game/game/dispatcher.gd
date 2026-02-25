extends Node

# central signals
signal scored(side)
signal ball_destroyed

func emit_scored(side: String):
	# any node can call this to broadcast
	emit_signal("scored", side)

func emit_ball_destroyed():	
	emit_signal("ball_destroyed")
	
