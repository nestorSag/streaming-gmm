trait Prior{
	def evaluate(): Double
	def getGradient(): List((Matrix,Double))
}