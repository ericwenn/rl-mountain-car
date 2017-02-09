import requests
import datetime



class Firebase(object):
	def __init__(self):
		self.firebaseUrl = "https://neural-networks-fe997.firebaseio.com/"
		self.firebaseSecret = "5SoQ1BSdV0W1bhH91hjEGpqrzJShgLUEU28FuGWq"


	def url( self, endpoint ):
		return "{}{}?auth={}".format(self.firebaseUrl, endpoint, self.firebaseSecret)

	def createSolution( self, problem_name, config ):

		self.addProblem(problem_name)


		res = requests.post( self.url("solutions.json") , json={
			"problem_name": problem_name,
			"config": config,
			"created_at": datetime.datetime.utcnow().isoformat(),
			"updated_at": datetime.datetime.utcnow().isoformat() 
		})

		solution_id = res.json()['name']

		self.pushRewardsToSolution(solution_id, [])

		requests.put( self.url("solutions/{}/id.json".format(solution_id)), json=solution_id)

		return solution_id


	def addProblem( self, problem_name ):
		res = requests.get( self.url( "problems.json" ))

		if( res.json() == None):
			problems = [problem_name]
		else:
			problems = set(res.json()).add(problem_name)

		requests.put( self.url( "problems.json"), json=problems)



	def pushRewardsToSolution( self, solution_id, rewards ):

		res = requests.get( self.url( "rewards/{}.json".format(solution_id)))

		if( not res.json() == None):
			rewards = list(res.json()) + rewards	
		
		res = requests.put( self.url( "rewards/{}.json".format(solution_id)), json=rewards)

